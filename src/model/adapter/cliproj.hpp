#ifndef __SD_MODEL_ADAPTER_CLIPROJ_HPP__
#define __SD_MODEL_ADAPTER_CLIPROJ_HPP__

#include "core/ggml_extend.hpp"
#include "model_loader.h"
#include "model_manager.h"

// ClipProj replaces the large text encoder of a diffusion model with a small
// one plus a learned linear projection into the large encoder's conditioning
// space. The published matrices target MiniMax-H3: a Qwen3-VL-4B (2560 dims)
// or 8B (4096 dims) hidden state, taken at one layer ("tap"), is mapped to the
// 5120-dim conditioning the DiT expects:
//
//     cond = (((h - mean_in) / std_in) @ W + mlp((h - mean_in) / std_in))
//            * std_out + mean_out
//
// plus the attention-sink vector substituted at token 0. The file is a plain
// .safetensors holding W, mean_in, std_in, mean_out, std_out, sink_out,
// mlp.0.weight/bias, mlp.2.weight/bias and the scalar metadata "tap" (layer
// index of the small encoder to read). The v3 matrices dropped W after the
// residual network subsumed the ridge fit, and widened mlp.0 to 32768 hidden
// units; both variants are detected from the file.
namespace ClipProj {

// The projection file stores W in PyTorch layout [d_in, d_out]. After the
// safetensors reader reverses the axes the ggml shape is [d_out, d_in], so the
// weight is declared transposed and materialised contiguously in the forward
// pass, mirroring CLIPProjection's transpose_weight handling.
    struct ClipProj : public GGMLBlock {
    public:
        int64_t d_in;
        int64_t d_out;
        int64_t mlp_hidden;
        bool has_ridge = false;

        ClipProj(int64_t d_in_, int64_t d_out_, int64_t mlp_hidden_)
            : d_in(d_in_), d_out(d_out_), mlp_hidden(mlp_hidden_) {
            blocks["mlp.0"] = std::make_shared<Linear>(d_in, mlp_hidden, true);
            blocks["mlp.2"] = std::make_shared<Linear>(mlp_hidden, d_out, true);
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            if (tensor_storage_map.find(prefix + "W") != tensor_storage_map.end()) {
                has_ridge       = true;
                enum ggml_type wtype = get_type(prefix + "W", tensor_storage_map, GGML_TYPE_F32);
                if (d_in % ggml_blck_size(wtype) != 0) {
                    wtype = GGML_TYPE_F32;
                }
                params["W"] = ggml_new_tensor_2d(ctx, wtype, d_out, d_in);
            }
            params["mean_in"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_in);
            params["std_in"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_in);
            params["mean_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_out);
            params["std_out"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_out);
            params["sink_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_out);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [d_in, n_token]
            auto mlp0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            // Standardised space, where the ridge was fitted.
            auto xn = ggml_div(ctx->ggml_ctx,
                               ggml_sub(ctx->ggml_ctx, x, params["mean_in"]),
                               params["std_in"]);  // [d_in, n_token]

            // The residual network keeps the dtype it was saved in (fp16 in the
            // published matrices); inputs and outputs are converted around it.
            auto res = mlp2->forward(ctx,
                                     ggml_gelu(ctx->ggml_ctx,
                                               mlp0->forward(ctx,
                                                             ggml_cast(ctx->ggml_ctx, xn, GGML_TYPE_F16))));
            ggml_tensor* sum = ggml_cast(ctx->ggml_ctx, res, GGML_TYPE_F32);
            if (has_ridge) {
                auto w  = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, params["W"]));
                auto yn = ggml_ext_linear(ctx->ggml_ctx, xn, w, nullptr);  // [d_out, n_token]
                sum     = ggml_add(ctx->ggml_ctx, yn, sum);
            }
            auto cond = ggml_add(ctx->ggml_ctx,
                                 ggml_mul(ctx->ggml_ctx, sum, params["std_out"]),
                                 params["mean_out"]);  // [d_out, n_token]

            // Token 0 is an attention sink: its direction is constant across
            // prompts, so substituting its measured value is exact rather than
            // approximate. The calibration excluded it from the fit because its
            // extreme norm would wreck the statistics.
            if (x->ne[1] > 1) {
                auto head = ggml_reshape_2d(ctx->ggml_ctx, params["sink_out"], d_out, 1);
                auto tail = ggml_ext_slice(ctx->ggml_ctx, cond, 1, 1, x->ne[1]);
                return ggml_concat(ctx->ggml_ctx, head, tail, 1);
            }
            return ggml_reshape_2d(ctx->ggml_ctx, params["sink_out"], d_out, 1);
        }
    };

    struct ClipProjRunner : public GGMLRunner {
        ClipProj model;
        int tap = 24;
        bool enabled = true;

        struct Dims {
            int64_t d_in;
            int64_t d_out;
            int64_t mlp_hidden;
        };

        static Dims infer_dims(const String2TensorStorage& tensor_storage_map,
                               const std::string& prefix) {
            auto mlp0 = tensor_storage_map.find(prefix + ".mlp.0.weight");
            auto mlp2 = tensor_storage_map.find(prefix + ".mlp.2.weight");
            if (mlp0 == tensor_storage_map.end() || mlp2 == tensor_storage_map.end()) {
                LOG_ERROR("cliproj: projection file has no '%s.mlp.0.weight' or '%s.mlp.2.weight' tensor",
                          prefix.c_str(), prefix.c_str());
                return {0, 0, 0};
            }
            // The file stores weights as torch [rows, cols]; the safetensors
            // reader reverses the axes, so ggml ne[0] is the torch last dim.
            auto w = tensor_storage_map.find(prefix + ".W");
            return {
                w != tensor_storage_map.end() ? w->second.ne[1] : mlp0->second.ne[0],
                w != tensor_storage_map.end() ? w->second.ne[0] : mlp2->second.ne[1],
                mlp0->second.ne[1],
            };
        }

    public:
        ClipProjRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map,
                       const std::string prefix,
                       int tap_,
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : ClipProjRunner(backend, tensor_storage_map, prefix, tap_, weight_manager,
                             infer_dims(tensor_storage_map, prefix)) {
        }

        std::string get_desc() override {
            return "cliproj";
        }

        int get_tap() const { return tap; }

        int64_t get_d_in() const { return model.d_in; }

        bool is_enabled() const { return enabled; }

        void set_enabled(bool enabled_) { enabled = enabled_; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void get_param_tensor_ops(std::map<ggml_tensor*, enum ggml_op>& tensor_ops) {
            model.get_param_tensor_ops(tensor_ops);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
            ggml_cgraph* gf = ggml_new_graph(compute_ctx);
            auto x          = make_input(x_tensor);
            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx, x);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  bool auto_free           = true,
                                  bool free_compute_buffer = true,
                                  bool free_compute_params = true) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph,
                                                                              n_threads,
                                                                              auto_free,
                                                                              free_compute_buffer,
                                                                              free_compute_params),
                                                   2);
        }

    private:
        ClipProjRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map,
                       const std::string prefix,
                       int tap_,
                       std::shared_ptr<RunnerWeightManager> weight_manager,
                       Dims dims)
            : GGMLRunner(backend, weight_manager),
              tap(tap_),
              model(dims.d_in, dims.d_out, dims.mlp_hidden) {
            model.init(params_ctx, tensor_storage_map, prefix);
            LOG_INFO("cliproj: d_in=%lld d_out=%lld mlp_hidden=%lld ridge=%d tap=%d",
                     static_cast<long long>(model.d_in),
                     static_cast<long long>(model.d_out),
                     static_cast<long long>(model.mlp_hidden),
                     model.has_ridge ? 1 : 0,
                     tap);
        }
    };

}  // namespace ClipProj

#endif  // __SD_MODEL_ADAPTER_CLIPROJ_HPP__

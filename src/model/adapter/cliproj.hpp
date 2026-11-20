#ifndef __SD_MODEL_ADAPTER_CLIPROJ_HPP__
#define __SD_MODEL_ADAPTER_CLIPROJ_HPP__

#include <map>
#include <memory>
#include <string>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
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
// plus the attention-sink vector substituted at token 0. The file is a
// .safetensors (or a GGUF with the "clipproj" architecture) holding mean_in,
// std_in, mean_out, std_out and the scalar "tap" (layer index of the small
// encoder to read). W (ridge matrix), the residual network mlp.0/mlp.2
// weight/bias and sink_out are each optional: the plain ridge files carry W
// and sink_out only, the v3 matrices dropped W after the residual network
// subsumed the ridge fit and widened mlp.0 to 32768 hidden units, and v3.1
// "-mlp" additionally dropped the measured sink vector. All variants are
// detected from the file, with at least one of W or the residual network.
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
        bool has_sink   = false;

        ClipProj(int64_t d_in_, int64_t d_out_, int64_t mlp_hidden_)
            : d_in(d_in_), d_out(d_out_), mlp_hidden(mlp_hidden_) {
            if (mlp_hidden > 0) {
                blocks["mlp.0"] = std::make_shared<Linear>(d_in, mlp_hidden, true);
                blocks["mlp.2"] = std::make_shared<Linear>(mlp_hidden, d_out, true);
            }
        }

        bool has_residual() const { return mlp_hidden > 0; }

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
            if (tensor_storage_map.find(prefix + "sink_out") != tensor_storage_map.end()) {
                has_sink        = true;
                params["sink_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_out);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [d_in, n_token]
            // Standardised space, where the ridge was fitted.
            auto xn = ggml_div(ctx->ggml_ctx,
                               ggml_sub(ctx->ggml_ctx, x, params["mean_in"]),
                               params["std_in"]);  // [d_in, n_token]

            ggml_tensor* sum = nullptr;
            if (has_residual()) {
                // The residual network keeps the dtype it was saved in (fp16 in the
                // published matrices); inputs and outputs are converted around it.
                auto mlp0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
                auto mlp2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
                auto res  = mlp2->forward(ctx,
                                          ggml_gelu(ctx->ggml_ctx,
                                                    mlp0->forward(ctx,
                                                                  ggml_cast(ctx->ggml_ctx, xn, GGML_TYPE_F16))));
                sum       = ggml_cast(ctx->ggml_ctx, res, GGML_TYPE_F32);
            }
            if (has_ridge) {
                auto w  = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, params["W"]));
                auto yn = ggml_ext_linear(ctx, xn, w, nullptr);  // [d_out, n_token]
                sum     = sum == nullptr ? yn : ggml_add(ctx->ggml_ctx, yn, sum);
            }
            GGML_ASSERT(sum != nullptr);
            auto cond = ggml_add(ctx->ggml_ctx,
                                 ggml_mul(ctx->ggml_ctx, sum, params["std_out"]),
                                 params["mean_out"]);  // [d_out, n_token]

            // Token 0 is an attention sink: its direction is constant across
            // prompts, so substituting its measured value is exact rather than
            // approximate. The calibration excluded it from the fit because its
            // extreme norm would wreck the statistics. The v3.1 matrices ship
            // without the measured sink and skip the substitution.
            if (!has_sink) {
                return cond;
            }
            if (x->ne[1] > 1) {
                auto head = ggml_reshape_2d(ctx->ggml_ctx, params["sink_out"], d_out, 1);
                auto tail = ggml_ext_slice(ctx, cond, 1, 1, x->ne[1]);
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
            // The file stores weights as torch [rows, cols]; the readers reverse the
            // axes, so ggml ne[0] is the torch last dim. mean_in / mean_out carry
            // d_in / d_out in every variant, with or without W.
            auto mean_in  = tensor_storage_map.find(prefix + ".mean_in");
            auto mean_out = tensor_storage_map.find(prefix + ".mean_out");
            for (const char* name : {"std_in", "std_out"}) {
                if (tensor_storage_map.find(prefix + "." + name) == tensor_storage_map.end()) {
                    LOG_ERROR("cliproj: projection file has no '%s.%s' tensor", prefix.c_str(), name);
                    return {0, 0, 0};
                }
            }
            if (mean_in == tensor_storage_map.end() || mean_out == tensor_storage_map.end()) {
                LOG_ERROR("cliproj: projection file has no '%s.mean_in' or '%s.mean_out' tensor",
                          prefix.c_str(), prefix.c_str());
                return {0, 0, 0};
            }
            const int64_t d_in  = mean_in->second.ne[0];
            const int64_t d_out = mean_out->second.ne[0];

            auto w    = tensor_storage_map.find(prefix + ".W");
            auto mlp0 = tensor_storage_map.find(prefix + ".mlp.0.weight");
            auto mlp2 = tensor_storage_map.find(prefix + ".mlp.2.weight");
            const bool has_mlp0 = mlp0 != tensor_storage_map.end();
            const bool has_mlp2 = mlp2 != tensor_storage_map.end();
            if (has_mlp0 != has_mlp2) {
                LOG_ERROR("cliproj: projection file has only one of '%s.mlp.0.weight' and '%s.mlp.2.weight'",
                          prefix.c_str(), prefix.c_str());
                return {0, 0, 0};
            }
            if (w == tensor_storage_map.end() && !has_mlp0) {
                LOG_ERROR("cliproj: projection file has neither '%s.W' nor a residual network", prefix.c_str());
                return {0, 0, 0};
            }
            if (w != tensor_storage_map.end() && (w->second.ne[1] != d_in || w->second.ne[0] != d_out)) {
                LOG_ERROR("cliproj: W is [%lld, %lld] but mean_in/mean_out say [%lld, %lld]",
                          static_cast<long long>(w->second.ne[1]), static_cast<long long>(w->second.ne[0]),
                          static_cast<long long>(d_in), static_cast<long long>(d_out));
                return {0, 0, 0};
            }
            int64_t mlp_hidden = 0;
            if (has_mlp0) {
                mlp_hidden = mlp0->second.ne[1];
                if (mlp0->second.ne[0] != d_in || mlp2->second.ne[0] != mlp_hidden || mlp2->second.ne[1] != d_out) {
                    LOG_ERROR("cliproj: residual network shapes disagree with mean_in/mean_out");
                    return {0, 0, 0};
                }
            }
            return {d_in, d_out, mlp_hidden};
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
                                  bool auto_runner_end = true) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, auto_runner_end), 2);
        }

    private:
        ClipProjRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map,
                       const std::string prefix,
                       int tap_,
                       std::shared_ptr<RunnerWeightManager> weight_manager,
                       Dims dims)
            : GGMLRunner(backend, weight_manager),
              model(dims.d_in, dims.d_out, dims.mlp_hidden),
              tap(tap_) {
            if (dims.d_in <= 0 || dims.d_out <= 0) {
                enabled = false;
                return;
            }
            model.init(params_ctx, tensor_storage_map, prefix);
            LOG_INFO("cliproj: d_in=%lld d_out=%lld mlp_hidden=%lld ridge=%d sink=%d tap=%d",
                     static_cast<long long>(model.d_in),
                     static_cast<long long>(model.d_out),
                     static_cast<long long>(model.mlp_hidden),
                     model.has_ridge ? 1 : 0,
                     model.has_sink ? 1 : 0,
                     tap);
        }
    };

}  // namespace ClipProj

#endif  // __SD_MODEL_ADAPTER_CLIPROJ_HPP__

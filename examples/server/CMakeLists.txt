set(TARGET sd-server)

add_executable(${TARGET}
    ../common/common.cpp
    ../common/generation_metadata.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    ../common/media_metadata.cpp
    main.cpp
    runtime.cpp
    model_catalog.cpp
    routes_models.cpp
    async_jobs.cpp
    routes_index.cpp
    routes_openai.cpp
    routes_sdapi.cpp
    routes_sdcpp.cpp
)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()

file(GLOB BUILTIN_RECIPE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/recipes/*.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${BUILTIN_RECIPE_FILES})
# Keep recipes in separate literals to stay within MSVC's string size limit.
set(SD_BUILTIN_RECIPES "")
foreach(RECIPE_FILE IN LISTS BUILTIN_RECIPE_FILES)
    file(READ "${RECIPE_FILE}" RECIPE_JSON)
    string(APPEND SD_BUILTIN_RECIPES "R\"recipes(${RECIPE_JSON})recipes\",\n")
endforeach()
configure_file(recipes_builtin.h.in "${CMAKE_CURRENT_BINARY_DIR}/recipes_builtin.h" @ONLY)
target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
endif()

# due to httplib; it contains a pragma for MSVC, but other things need explicit flags
if(WIN32 AND NOT MSVC)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

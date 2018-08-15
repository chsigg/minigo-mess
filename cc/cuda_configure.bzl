"""Build rule generator for locally installed CUDA toolkit."""

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    cuda_path = _get_env_var(repository_ctx, "CUDA_PATH", "/usr/local/cuda")

    print("Using CUDA from %s\n" % cuda_path)

    repository_ctx.symlink(cuda_path, "cuda")

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda",
    hdrs = glob(
        include = ["cuda/include/**/*.h*"],
    ),
    strip_include_prefix = "cuda/include",
    include_prefix = "cuda",
    srcs = [
         "cuda/lib64/stubs/libcuda.so",
         "cuda/lib64/libcudart_static.a",
    ],
    linkopts = ["-ldl", "-lrt"],
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = ["CUDA_PATH", "CUDNN_PATH"],
)

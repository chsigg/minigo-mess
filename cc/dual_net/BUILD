package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_binary",
    "minigo_cc_library",
    "minigo_cc_test",
    "minigo_cc_test_19_only",
    "minigo_cc_test_9_only",
)

config_setting(
    name = "enable_remote",
    define_values = {"remote": "1"},
)

config_setting(
    name = "enable_tf",
    define_values = {"tf": "1"},
)

config_setting(
    name = "enable_lite",
    define_values = {"lite": "1"},
)

config_setting(
    name = "enable_trt",
    define_values = {"trt": "1"},
)

minigo_cc_library(
    name = "dual_net",
    srcs = ["dual_net.cc"],
    hdrs = ["dual_net.h"],
    deps = [
        "//cc:base",
        "//cc:position",
        "@com_google_absl//absl/types:span",
        "@com_github_gflags_gflags//:gflags",
    ],
)

factory_engine_copts = select({
    ":enable_remote": ["-DMG_ENABLE_REMOTE_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    ":enable_tf": ["-DMG_ENABLE_TF_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    ":enable_lite": ["-DMG_ENABLE_LITE_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    ":enable_trt": ["-DMG_ENABLE_TRT_DUAL_NET"],
    "//conditions:default": [],
})

factory_engine_deps = select({
    ":enable_remote": [":remote_dual_net"],
    "//conditions:default": [],
}) + select({
    ":enable_tf": [":tf_dual_net"],
    "//conditions:default": [],
}) + select({
    ":enable_lite": [":lite_dual_net"],
    "//conditions:default": [],
}) + select({
    ":enable_trt": [":trt_dual_net"],
    "//conditions:default": [],
})

minigo_cc_library(
    name = "factory",
    srcs = ["factory.cc"],
    hdrs = ["factory.h"],
    copts = factory_engine_copts,
    deps = [
        ":dual_net",
        ":batching_dual_net",
        "//cc:base",
        "//cc:check",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + factory_engine_deps,
)

minigo_cc_library(
    name = "fake_net",
    testonly = 1,
    srcs = ["fake_dual_net.cc"],
    hdrs = ["fake_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
    ],
)

minigo_cc_library(
    name = "remote_dual_net",
    srcs = ["remote_dual_net.cc"],
    hdrs = ["remote_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:thread_safe_queue",
        "//proto:inference_service_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_grpc//:grpc++",
    ],
)

minigo_cc_library(
    name = "tf_dual_net",
    srcs = ["tf_dual_net.cc"],
    hdrs = ["tf_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:tensorflow",
        "//cc:thread_safe_queue",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "lite_dual_net",
    srcs = ["lite_dual_net.cc"],
    hdrs = ["lite_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:tf_lite",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "trt_dual_net",
    srcs = ["trt_dual_net.cc"],
    hdrs = ["trt_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:tensorrt",
        "//cc:thread_safe_queue",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//:cuda",
    ],
)

minigo_cc_library(
    name = "batching_dual_net",
    srcs = ["batching_service.cc"],
    hdrs = ["batching_service.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_test_9_only(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    deps = [
        ":dual_net",
        "//cc:position",
        "//cc:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "inference_server_test",
    srcs = ["remote_dual_net_test.cc"],
    deps = [
        ":fake_net",
        ":remote_dual_net",
        "//cc:random",
        "//proto:inference_service_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

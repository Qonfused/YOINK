load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

py_library(
    name = "yoink",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
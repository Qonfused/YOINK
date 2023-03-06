workspace(name = "project_yoink")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# FIXME: `py_repositories` is a no-op and is depreciated.
# load("@rules_python//python:repositories.bzl", "py_repositories")
# py_repositories()

#
# Direct dependencies
#

git_repository(
    name = "com_google_absl",
    commit = "66665d8d2e3fedff340b83f9841ca427145a7b26",
    remote = "https://github.com/abseil/abseil-cpp.git",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

git_repository(
    name = "rules_license",
    remote = "https://github.com/bazelbuild/rules_license.git",
    tag = "0.0.4",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.5.0",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_abseil
new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v2.9.2",
)

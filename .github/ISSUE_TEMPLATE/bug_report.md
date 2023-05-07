---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/abetlen/llama-cpp-python/blob/main/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/abetlen/llama-cpp-python/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama-cpp-python` to do.

# Current Behavior

Please provide a detailed written description of what `llama-cpp-python` did, instead.

# Environment and Context

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

* Physical (or virtual) hardware you are using, e.g. for Linux:

`$ lscpu`

* Operating System, e.g. for Linux:

`$ uname -a`

* SDK version, e.g. for Linux:

```
$ python3 --version
$ make --version
$ g++ --version
```

# Failure Information (for bugs)

Please help provide information about the failure if this is a bug. If it is not a bug, please remove the rest of this template.

# Steps to Reproduce

Please provide detailed steps for reproducing the issue. We are not sitting in front of your screen, so the more detail the better.

1. step 1
2. step 2
3. step 3
4. etc.

**Note: Many issues seem to be regarding performance issues / differences with `llama.cpp`. In these cases we need to confirm that you're comparing against the version of `llama.cpp` that was built with your python package, and which parameters you're passing to the context.**

# Failure Logs

Please include any relevant log snippets or files. If it works under one configuration but not under another, please provide logs for both configurations and their corresponding outputs so it is easy to see where behavior changes.

Also, please try to **avoid using screenshots** if at all possible. Instead, copy/paste the console output and use [Github's markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to cleanly format your logs for easy readability.

Example environment info:
```
llama-cpp-python$ git log | head -1
commit 47b0aa6e957b93dbe2c29d53af16fbae2dd628f2

llama-cpp-python$ python3 --version
Python 3.10.10

llama-cpp-python$ pip list | egrep "uvicorn|fastapi|sse-starlette"
fastapi            0.95.0
sse-starlette      1.3.3
uvicorn            0.21.1
```

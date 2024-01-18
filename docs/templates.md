# Templates

This document provides a comprehensive guide to the integration of Jinja2 templating into the `llama-cpp-python` project, with a focus on enhancing the chat functionality of the `llama-2` model.

## Introduction

- Brief explanation of the `llama-cpp-python` project's need for a templating system.
- Overview of the `llama-2` model's interaction with templating.

## Jinja2 Dependency Integration

- Rationale for choosing Jinja2 as the templating engine.
  - Compatibility with Hugging Face's `transformers`.
  - Desire for advanced templating features and simplicity.
- Detailed steps for adding `jinja2` to `pyproject.toml` for dependency management.

## Template Management Refactor

- Summary of the refactor and the motivation behind it.
- Description of the new chat handler selection logic:
  1. Preference for a user-specified `chat_handler`.
  2. Fallback to a user-specified `chat_format`.
  3. Defaulting to a chat format from a `.gguf` file if available.
  4. Utilizing the `llama2` default chat format as the final fallback.
- Ensuring backward compatibility throughout the refactor.

## Implementation Details

- In-depth look at the new `AutoChatFormatter` class.
- Example code snippets showing how to utilize the Jinja2 environment and templates.
- Guidance on how to provide custom templates or use defaults.

## Testing and Validation

- Outline of the testing strategy to ensure seamless integration.
- Steps for validating backward compatibility with existing implementations.

## Benefits and Impact

- Analysis of the expected benefits, including consistency, performance gains, and improved developer experience.
- Discussion of the potential impact on current users and contributors.

## Future Work

- Exploration of how templating can evolve within the project.
- Consideration of additional features or optimizations for the templating engine.
- Mechanisms for community feedback on the templating system.

## Conclusion

- Final thoughts on the integration of Jinja2 templating.
- Call to action for community involvement and feedback.

# Security

Do not commit API keys, tokens, credentials, private datasets, model weights, or absolute machine-local paths. LLM access is optional and should use an environment variable such as `DEEPSEEK_API_KEY`.

If a credential has been committed, revoke it at the provider immediately, remove it from the working tree, and rotate it before publishing any Git history. Please report security issues privately to the repository owner rather than opening a public issue with the secret.

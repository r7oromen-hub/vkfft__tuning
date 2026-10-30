"""Read only the relevant local Codex configuration keys, without secrets."""
import json
import os
import tomllib
from pathlib import Path

root = Path(os.environ['USERPROFILE']) / '.codex'
paths = [root/'config.toml', Path.cwd()/'.codex/config.toml', root/'requirements.toml']
result = []
for path in paths:
    entry = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding='utf8'))
            agents = data.get('agents', {})
            features = data.get('features', {})
            entry.update(agents_enabled=agents.get('enabled'),
                         max_concurrent=agents.get('max_concurrent_threads_per_session'),
                         legacy_max_threads=agents.get('max_threads'),
                         features_multi_agent=features.get('multi_agent'))
        except (OSError, ValueError, TypeError) as exc:
            entry['error'] = type(exc).__name__
    result.append(entry)
print(json.dumps(result, indent=2))

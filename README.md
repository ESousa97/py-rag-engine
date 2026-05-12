# py-rag-engine

RAG engine com ingestão de documentos (PDF, Markdown).

## Visão geral

Projeto simples para demonstrar ingestão de documentos e preparação para Retrieval-Augmented Generation (RAG).

## Quickstart

1. Crie um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale o pacote e dependências de teste:

```powershell
pip install -e .
pip install pytest
```

3. Rode os testes:

```powershell
python -m pytest -q
```

## Desenvolvimento

- Código fonte em `internal/`.
- Testes em `tests/`.

## Contribuindo

Abra um pull request com descrição do que mudou e garanta que os testes passam.

## Licença

Este repositório está licenciado sob a licença MIT. Veja o arquivo `LICENSE`.

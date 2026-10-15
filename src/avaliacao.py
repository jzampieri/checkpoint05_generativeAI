"""Prompt, geração e métricas de avaliação (execução do SQL no SQLite)."""
import re
import sqlite3
import time

import torch

from .banco import ESQUEMA_SQL

INSTRUCAO_SISTEMA = (
    "Você é um assistente que converte perguntas em português para consultas SQL (SQLite) "
    "sobre o banco de dados de uma loja virtual. Responda apenas com a consulta SQL, sem explicações.\n\n"
    "Esquema do banco:\n" + ESQUEMA_SQL + "\n\n"
    "Valores possíveis de pedidos.status: 'entregue', 'enviado', 'pendente', 'cancelado'. "
    "Categorias: 'Eletrônicos', 'Livros', 'Roupas', 'Casa', 'Esportes', 'Brinquedos'. "
    "Estados usam a sigla (ex.: 'SP'). Datas no formato 'AAAA-MM-DD'."
)


def mensagens(pergunta: str, sql: str | None = None):
    msgs = [
        {"role": "system", "content": INSTRUCAO_SISTEMA},
        {"role": "user", "content": pergunta},
    ]
    if sql is not None:
        msgs.append({"role": "assistant", "content": sql})
    return msgs


def prompt_texto(tokenizer, pergunta: str) -> str:
    return tokenizer.apply_chat_template(mensagens(pergunta), tokenize=False, add_generation_prompt=True)


def limpar_sql(texto: str) -> str:
    """Extrai o SQL da resposta do modelo (remove cercas markdown e texto extra)."""
    texto = texto.strip()
    bloco = re.search(r"```(?:sql)?\s*(.*?)(```|$)", texto, flags=re.S | re.I)
    if bloco:
        texto = bloco.group(1).strip()
    if ";" in texto:
        texto = texto[: texto.index(";") + 1]
    return texto.strip()


@torch.no_grad()
def gerar_sql(modelo, tokenizer, perguntas, max_new_tokens=128, batch_size=16, do_sample=False,
              temperature=1.0, top_p=1.0):
    """Gera SQL para uma lista de perguntas (em lotes, com padding à esquerda)."""
    tokenizer.padding_side = "left"
    saidas = []
    modelo.eval()
    for i in range(0, len(perguntas), batch_size):
        lote = [prompt_texto(tokenizer, p) for p in perguntas[i: i + batch_size]]
        enc = tokenizer(lote, return_tensors="pt", padding=True).to(modelo.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, do_sample=do_sample)
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        out = modelo.generate(**enc, **kwargs)
        novos = out[:, enc["input_ids"].shape[1]:]
        saidas += [limpar_sql(t) for t in tokenizer.batch_decode(novos, skip_special_tokens=True)]
    return saidas


def executar(con: sqlite3.Connection, sql: str):
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:  # SQL inválido
        return None, str(e)


def _normalizar(linhas, ordenado: bool):
    def norm(v):
        return round(v, 2) if isinstance(v, float) else v
    linhas = [tuple(norm(v) for v in linha) for linha in linhas]
    return linhas if ordenado else sorted(linhas, key=repr)


def mesma_resposta(con, sql_pred: str, sql_ouro: str) -> tuple[bool, bool]:
    """Retorna (sql_valido, resultado_correto) comparando os resultados da execução."""
    res_pred, erro = executar(con, sql_pred)
    if erro is not None:
        return False, False
    res_ouro, _ = executar(con, sql_ouro)
    ordenado = "order by" in sql_ouro.lower()
    return True, _normalizar(res_pred, ordenado) == _normalizar(res_ouro, ordenado)


def _norm_texto(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def avaliar(modelo, tokenizer, exemplos, con, **kwargs_geracao):
    """Gera e avalia. Retorna (métricas, lista de predições)."""
    t0 = time.time()
    preds = gerar_sql(modelo, tokenizer, [e["pergunta"] for e in exemplos], **kwargs_geracao)
    duracao = time.time() - t0
    registros = []
    for e, p in zip(exemplos, preds):
        valido, correto = mesma_resposta(con, p, e["sql"])
        registros.append({**e, "pred": p, "valido": valido, "correto": correto,
                          "exato": _norm_texto(p) == _norm_texto(e["sql"])})
    n = len(registros)
    metricas = {
        "acuracia_execucao": sum(r["correto"] for r in registros) / n,
        "match_exato": sum(r["exato"] for r in registros) / n,
        "sql_valido": sum(r["valido"] for r in registros) / n,
        "tempo_geracao_s": duracao,
    }
    return metricas, registros

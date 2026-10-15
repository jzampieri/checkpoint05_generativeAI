"""Frontend (Gradio) para fazer perguntas em português e obter o SQL gerado pela LLM treinada.

Uso:  .venv\\Scripts\\python app.py   ->  abre em http://127.0.0.1:7860
"""
import json
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.avaliacao import limpar_sql, prompt_texto
from src.banco import ESQUEMA_SQL, conectar

RAIZ = Path(__file__).resolve().parent
PASTA_MODELOS = RAIZ / "modelos"
SEM_FT = " (sem fine-tuning)"

if not (PASTA_MODELOS / "melhor").exists():
    raise SystemExit("Modelo não encontrado. Execute antes o notebook notebooks/treinamento_text2sql.ipynb.")
assert torch.cuda.is_available(), "Este app foi feito para rodar na GPU (CUDA)."

INFO = json.loads((PASTA_MODELOS / "melhor" / "info.json").read_text(encoding="utf-8"))
RESULTADOS = {}
if (RAIZ / "resultados" / "resultados.json").exists():
    RESULTADOS = json.loads((RAIZ / "resultados" / "resultados.json").read_text(encoding="utf-8"))["experimentos"]

# adaptadores LoRA treinados pelo notebook: nome -> (pasta, modelo base)
ADAPTADORES = {}
for pasta in sorted(PASTA_MODELOS.iterdir()):
    cfg = pasta / "adapter_config.json"
    if pasta.name != "melhor" and cfg.exists():
        ADAPTADORES[pasta.name] = (pasta, json.loads(cfg.read_text(encoding="utf-8"))["base_model_name_or_path"])
BASES = sorted({base for _, base in ADAPTADORES.values()})
OPCOES = [INFO["experimento"]] + [n for n in ADAPTADORES if n != INFO["experimento"]] + [b + SEM_FT for b in BASES]

_cache = {}  # modelo base -> (tokenizer, PeftModel com todos os adaptadores daquele base)


def nome_adaptador(nome):
    return nome.replace(".", "_")  # o PEFT não aceita "." no nome do adaptador


def carregar(base_id):
    if base_id not in _cache:
        tok = AutoTokenizer.from_pretrained(base_id)
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda")
        modelo = None
        for nome, (pasta, b) in ADAPTADORES.items():
            if b != base_id:
                continue
            if modelo is None:
                modelo = PeftModel.from_pretrained(base, pasta, adapter_name=nome_adaptador(nome))
            else:
                modelo.load_adapter(pasta, adapter_name=nome_adaptador(nome))
        _cache[base_id] = (tok, modelo.eval())
    return _cache[base_id]


carregar(INFO["modelo_base"])  # pré-carrega o modelo padrão

GPU = torch.cuda.get_device_properties(0)
INFO_GPU = f"{GPU.name} · {GPU.total_memory / 1024**3:.1f} GB VRAM · CUDA {torch.version.cuda} · PyTorch {torch.__version__}"

EXEMPLOS = [
    "Quais são os 5 produtos mais vendidos?",
    "Quantos pedidos foram cancelados?",
    "Qual o ticket médio dos pedidos por estado?",
    "Quais produtos da categoria Eletrônicos custam menos de 500 reais?",
    "Quantos clientes moram em Curitiba?",
    "Qual foi o faturamento dos pedidos entregues em 2024?",
    "Quais clientes nunca fizeram pedidos?",
]


def descricao(opcao):
    if opcao.endswith(SEM_FT):
        return f"Modelo **{opcao}** — LoRA desligado."
    r = RESULTADOS.get(opcao, {})
    extra = f" · acurácia validação {r['val_acuracia']:.1%} · teste {r['teste_acuracia']:.1%}" if r else ""
    padrao = " · *escolhido pelo critério de validação*" if opcao == INFO["experimento"] else ""
    return f"Adaptador **{opcao}** sobre `{ADAPTADORES[opcao][1]}`{extra}{padrao}"


@torch.no_grad()
def responder(pergunta, opcao, temperatura, top_p, max_tokens):
    if not pergunta.strip():
        return "", pd.DataFrame(), "Digite uma pergunta."
    sem_ft = opcao.endswith(SEM_FT)
    base_id = opcao.removesuffix(SEM_FT) if sem_ft else ADAPTADORES[opcao][1]
    tok, modelo = carregar(base_id)

    enc = tok([prompt_texto(tok, pergunta)], return_tensors="pt").to("cuda")
    kwargs = dict(max_new_tokens=int(max_tokens), pad_token_id=tok.pad_token_id)
    if temperatura > 0:
        kwargs.update(do_sample=True, temperature=float(temperatura), top_p=float(top_p))
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    torch.cuda.synchronize()
    t0 = time.time()
    if sem_ft:
        with modelo.disable_adapter():
            out = modelo.generate(**enc, **kwargs)
    else:
        modelo.set_adapter(nome_adaptador(opcao))
        out = modelo.generate(**enc, **kwargs)
    torch.cuda.synchronize()
    dt = time.time() - t0

    novos = out[0, enc["input_ids"].shape[1]:]
    sql = limpar_sql(tok.decode(novos, skip_special_tokens=True))
    status = (f"{opcao} · {len(novos)} tokens em {dt:.2f}s ({len(novos) / dt:.0f} tokens/s) · "
              f"VRAM alocada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if not sql.lower().lstrip().startswith(("select", "with")):
        return sql, pd.DataFrame(), status + "\n⚠️ Somente consultas SELECT são executadas."
    try:
        cur = conectar().execute(sql)  # conexão por chamada: o Gradio usa várias threads; banco aberto só para leitura
        colunas = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchmany(200), columns=colunas)
        return sql, df, status + f"\n✅ Consulta executada: {len(df)} linha(s) exibidas."
    except Exception as e:
        return sql, pd.DataFrame(), status + f"\n❌ Erro ao executar o SQL: {e}"


with gr.Blocks(title="Text-to-SQL Loja") as demo:
    gr.Markdown(
        "# 🛒 Pergunte ao banco da loja\n"
        "Escreva uma pergunta em português; a LLM local (fine-tuning com LoRA) gera o SQL e a consulta é executada no banco SQLite.\n\n"
        f"**GPU:** {INFO_GPU}"
    )
    with gr.Row():
        with gr.Column(scale=3):
            pergunta = gr.Textbox(label="Sua pergunta (prompt)", lines=3,
                                  placeholder="Ex.: Quais são os 5 clientes que mais gastaram?")
            with gr.Row():
                botao = gr.Button("Gerar SQL e consultar", variant="primary")
                limpar = gr.ClearButton(value="Limpar")
            gr.Examples(EXEMPLOS, inputs=pergunta)
        with gr.Column(scale=2):
            opcao = gr.Dropdown(OPCOES, value=OPCOES[0], label="Modelo / adaptador LoRA")
            desc = gr.Markdown(descricao(OPCOES[0]))
            temperatura = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperatura (0 = greedy)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p (apenas com temperatura > 0)")
            max_tokens = gr.Slider(32, 256, value=128, step=16, label="Máximo de tokens gerados")
    sql = gr.Code(label="SQL gerado", language="sql")
    status = gr.Textbox(label="Informações", lines=2, interactive=False)
    tabela = gr.Dataframe(label="Resultado da consulta")
    with gr.Accordion("Esquema do banco de dados", open=False):
        gr.Code(ESQUEMA_SQL, language="sql")

    opcao.change(descricao, opcao, desc)
    entradas = [pergunta, opcao, temperatura, top_p, max_tokens]
    saidas = [sql, tabela, status]
    botao.click(responder, entradas, saidas)
    pergunta.submit(responder, entradas, saidas)
    limpar.add([pergunta, sql, tabela, status])

if __name__ == "__main__":
    demo.launch(show_error=True)

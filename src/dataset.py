"""Geração do dataset sintético pergunta (PT-BR) -> SQL sobre o banco da loja.

Estratégia de divisão (para uma avaliação honesta):
  * Cada template tem 5 paráfrases. As paráfrases 0-2 vão para o treino; a paráfrase 3
    aparece SOMENTE na validação e a 4 SOMENTE no teste ("paráfrase inédita"). Assim a
    validação mede generalização do mesmo jeito que o teste (e não memorização).
  * Parte do teste usa paráfrases conhecidas, mas com valores (cidades, anos, nomes...)
    que não aparecem em nenhum exemplo de treino com o mesmo template ("valores novos").
  * Alguns templates inteiros ficam fora do treino ("tipo inédito"), para medir generalização.
"""
import json
import random
import sqlite3
from pathlib import Path

from .banco import CAMINHO_BANCO, CIDADES, PRODUTOS, conectar

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"

NOMES_UF = {"SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais", "PR": "Paraná",
            "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "BA": "Bahia", "PE": "Pernambuco",
            "CE": "Ceará", "DF": "Distrito Federal", "GO": "Goiás", "AM": "Amazonas", "PA": "Pará"}
STATUS_PLURAL = {"entregue": "entregues", "enviado": "enviados", "pendente": "pendentes", "cancelado": "cancelados"}
UFS = sorted({uf for _, uf in CIDADES})
CATEGORIAS = list(PRODUTOS)
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro"]


def _slots_base(rng, ctx):
    cidade, _ = rng.choice(CIDADES)
    uf = rng.choice(UFS)
    status = rng.choice(list(STATUS_PLURAL))
    mes = rng.randint(1, 12)
    a = rng.choice([50, 100, 150, 200, 300, 500])
    return {
        "cidade": cidade,
        "uf": uf,
        "estado_nome": NOMES_UF[uf],
        "cat": rng.choice(CATEGORIAS),
        "status": status,
        "status_pl": STATUS_PLURAL[status],
        "ano": rng.choice([2023, 2024, 2025]),
        "mes": f"{mes:02d}",
        "mes_nome": MESES[mes - 1],
        "n": rng.choice([3, 5, 10]),
        "valor": rng.choice([50, 100, 200, 300, 500, 1000, 1500, 2000]),
        "a": a,
        "b": a + rng.choice([100, 200, 500, 1000]),
        "estoque": rng.choice([5, 10, 20, 30, 50]),
        "nome": rng.choice(ctx["clientes"]),
        "produto": rng.choice(ctx["produtos"]),
    }


# Cada template: sql + 5 paráfrases (0-2 treino, 3 validação, 4 teste).
TEMPLATES = [
    {"id": "conta_clientes_cidade",
     "sql": "SELECT COUNT(*) FROM clientes WHERE cidade = '{cidade}';",
     "perguntas": ["Quantos clientes moram em {cidade}?", "Qual o número de clientes da cidade de {cidade}?",
                   "Conte os clientes cadastrados em {cidade}.", "Quantos clientes temos em {cidade}?",
                   "Me diga a quantidade de clientes que vivem em {cidade}."]},
    {"id": "clientes_estado",
     "sql": "SELECT nome FROM clientes WHERE estado = '{uf}';",
     "perguntas": ["Liste os nomes dos clientes do estado {uf}.", "Quais clientes são de {uf}?",
                   "Mostre os clientes que moram no estado {uf}.", "Nomes dos clientes com estado igual a {uf}.",
                   "Quem são os clientes localizados em {uf}?"]},
    {"id": "produtos_categoria_preco_menor",
     "sql": "SELECT nome, preco FROM produtos WHERE categoria = '{cat}' AND preco < {valor};",
     "perguntas": ["Quais produtos da categoria {cat} custam menos de {valor} reais?",
                   "Liste nome e preço dos produtos de {cat} com preço abaixo de {valor}.",
                   "Mostre os produtos {cat} mais baratos que R$ {valor}.",
                   "Produtos da categoria {cat} com preço inferior a {valor}.",
                   "Preciso dos itens de {cat} que custam menos que {valor} reais, com seus preços."]},
    {"id": "preco_medio_categoria",
     "sql": "SELECT AVG(preco) FROM produtos WHERE categoria = '{cat}';",
     "perguntas": ["Qual o preço médio dos produtos da categoria {cat}?", "Calcule a média de preço dos produtos de {cat}.",
                   "Quanto custa em média um produto de {cat}?", "Preço médio da categoria {cat}.",
                   "Me informe a média dos preços na categoria {cat}."]},
    {"id": "top_produtos_caros",
     "sql": "SELECT nome, preco FROM produtos ORDER BY preco DESC LIMIT {n};",
     "perguntas": ["Quais são os {n} produtos mais caros?", "Liste os {n} produtos de maior preço.",
                   "Mostre o top {n} produtos mais caros da loja.", "Os {n} produtos com preço mais alto.",
                   "Me mostre nome e preço dos {n} itens mais caros do catálogo."]},
    {"id": "top_produtos_baratos_categoria",
     "sql": "SELECT nome, preco FROM produtos WHERE categoria = '{cat}' ORDER BY preco ASC LIMIT {n};",
     "perguntas": ["Quais são os {n} produtos mais baratos da categoria {cat}?",
                   "Liste os {n} itens de menor preço em {cat}.", "Top {n} produtos mais baratos de {cat}.",
                   "Mostre os {n} produtos de {cat} com os preços mais baixos.",
                   "Na categoria {cat}, quais os {n} produtos mais em conta?"]},
    {"id": "produtos_estoque_baixo",
     "sql": "SELECT nome, estoque FROM produtos WHERE estoque < {estoque};",
     "perguntas": ["Quais produtos têm estoque menor que {estoque} unidades?",
                   "Liste os produtos com estoque abaixo de {estoque}.",
                   "Mostre nome e estoque dos produtos com menos de {estoque} unidades.",
                   "Produtos com estoque inferior a {estoque}.",
                   "Quais itens estão com menos de {estoque} unidades em estoque?"]},
    {"id": "conta_pedidos_status",
     "sql": "SELECT COUNT(*) FROM pedidos WHERE status = '{status}';",
     "perguntas": ["Quantos pedidos estão com status {status}?", "Quantos pedidos foram {status_pl}?",
                   "Conte os pedidos {status_pl}.", "Qual a quantidade de pedidos com status '{status}'?",
                   "Total de pedidos que estão {status_pl}."]},
    {"id": "faturamento_ano",
     "sql": "SELECT SUM(valor_total) FROM pedidos WHERE status = 'entregue' AND strftime('%Y', data_pedido) = '{ano}';",
     "perguntas": ["Qual foi o faturamento dos pedidos entregues em {ano}?",
                   "Some o valor total dos pedidos entregues no ano de {ano}.",
                   "Quanto a loja faturou com pedidos entregues em {ano}?",
                   "Receita total de pedidos entregues em {ano}.",
                   "Qual o valor somado das vendas entregues durante {ano}?"]},
    {"id": "pedidos_por_status",
     "sql": "SELECT status, COUNT(*) FROM pedidos GROUP BY status;",
     "perguntas": ["Quantos pedidos existem por status?", "Conte os pedidos agrupados por status.",
                   "Mostre a quantidade de pedidos em cada status.", "Distribuição de pedidos por status.",
                   "Para cada status, quantos pedidos temos?"]},
    {"id": "produtos_por_categoria",
     "sql": "SELECT categoria, COUNT(*) FROM produtos GROUP BY categoria;",
     "perguntas": ["Quantos produtos há em cada categoria?", "Conte os produtos por categoria.",
                   "Mostre o número de produtos de cada categoria.", "Quantidade de produtos agrupada por categoria.",
                   "Qual o total de itens cadastrados por categoria?"]},
    {"id": "clientes_por_estado",
     "sql": "SELECT estado, COUNT(*) FROM clientes GROUP BY estado ORDER BY COUNT(*) DESC;",
     "perguntas": ["Quantos clientes existem por estado, do maior para o menor?",
                   "Conte os clientes por estado em ordem decrescente.",
                   "Ranking de estados pelo número de clientes.",
                   "Mostre a quantidade de clientes de cada estado, ordenada do maior para o menor.",
                   "Quais estados têm mais clientes? Mostre a contagem de todos em ordem decrescente."]},
    {"id": "pedidos_do_cliente",
     "sql": "SELECT p.id, p.data_pedido, p.valor_total FROM pedidos p JOIN clientes c ON p.cliente_id = c.id WHERE c.nome = '{nome}';",
     "perguntas": ["Quais pedidos o cliente {nome} fez?", "Liste os pedidos de {nome} com data e valor.",
                   "Mostre id, data e valor total dos pedidos do cliente {nome}.",
                   "Pedidos realizados por {nome}.",
                   "Quero ver o histórico de pedidos de {nome}."]},
    {"id": "total_gasto_cliente",
     "sql": "SELECT SUM(p.valor_total) FROM pedidos p JOIN clientes c ON p.cliente_id = c.id WHERE c.nome = '{nome}';",
     "perguntas": ["Quanto o cliente {nome} gastou no total?", "Qual o valor total de pedidos de {nome}?",
                   "Some o valor de todos os pedidos de {nome}.", "Total gasto por {nome}.",
                   "Qual a soma das compras feitas por {nome}?"]},
    {"id": "top_clientes_gasto",
     "sql": "SELECT c.nome, SUM(p.valor_total) AS total FROM clientes c JOIN pedidos p ON p.cliente_id = c.id GROUP BY c.id ORDER BY total DESC LIMIT {n};",
     "perguntas": ["Quais são os {n} clientes que mais gastaram?", "Liste os {n} maiores clientes por valor gasto.",
                   "Top {n} clientes por total de compras.", "Mostre os {n} clientes com maior soma de pedidos.",
                   "Quem são os {n} clientes que mais compraram em valor?"]},
    {"id": "top_produtos_vendidos",
     "sql": "SELECT pr.nome, SUM(i.quantidade) AS total_vendido FROM itens_pedido i JOIN produtos pr ON i.produto_id = pr.id GROUP BY pr.id ORDER BY total_vendido DESC LIMIT {n};",
     "perguntas": ["Quais são os {n} produtos mais vendidos?", "Liste os {n} produtos com maior quantidade vendida.",
                   "Top {n} produtos mais vendidos em unidades.", "Mostre os {n} itens que mais venderam.",
                   "Quais {n} produtos tiveram mais unidades vendidas?"]},
    {"id": "receita_por_categoria",
     "sql": "SELECT pr.categoria, SUM(i.quantidade * i.preco_unitario) AS receita FROM itens_pedido i JOIN produtos pr ON i.produto_id = pr.id GROUP BY pr.categoria ORDER BY receita DESC;",
     "perguntas": ["Qual a receita de cada categoria de produto?", "Some a receita por categoria, da maior para a menor.",
                   "Mostre o faturamento por categoria em ordem decrescente.", "Receita total agrupada por categoria.",
                   "Quanto cada categoria gerou de receita, ordenado do maior para o menor?"]},
    {"id": "pedidos_mes_ano",
     "sql": "SELECT COUNT(*) FROM pedidos WHERE strftime('%Y-%m', data_pedido) = '{ano}-{mes}';",
     "perguntas": ["Quantos pedidos foram feitos em {mes_nome} de {ano}?", "Conte os pedidos de {mes}/{ano}.",
                   "Qual o número de pedidos realizados no mês {mes} de {ano}?", "Pedidos feitos em {mes_nome}/{ano}.",
                   "Quantas compras aconteceram em {mes_nome} de {ano}?"]},
    {"id": "pedidos_acima_valor",
     "sql": "SELECT id, valor_total FROM pedidos WHERE valor_total > {valor} ORDER BY valor_total DESC;",
     "perguntas": ["Quais pedidos têm valor total acima de {valor} reais?",
                   "Liste os pedidos com valor maior que {valor}, do maior para o menor.",
                   "Mostre id e valor dos pedidos acima de R$ {valor}.", "Pedidos com valor_total superior a {valor}.",
                   "Quais compras passaram de {valor} reais? Ordene pelo valor decrescente."]},
    {"id": "clientes_cadastrados_ano",
     "sql": "SELECT nome FROM clientes WHERE strftime('%Y', data_cadastro) = '{ano}';",
     "perguntas": ["Quais clientes se cadastraram em {ano}?", "Liste os clientes cadastrados no ano de {ano}.",
                   "Mostre os nomes dos clientes com cadastro em {ano}.", "Clientes que entraram em {ano}.",
                   "Quem fez cadastro na loja durante {ano}?"]},
    {"id": "clientes_sem_pedidos",
     "sql": "SELECT nome FROM clientes WHERE id NOT IN (SELECT cliente_id FROM pedidos);",
     "perguntas": ["Quais clientes nunca fizeram pedidos?", "Liste os clientes sem nenhum pedido.",
                   "Mostre os clientes que ainda não compraram.", "Clientes sem pedidos registrados.",
                   "Quem são os clientes que nunca realizaram uma compra?"]},
    {"id": "ticket_medio_estado",
     "sql": "SELECT c.estado, AVG(p.valor_total) FROM pedidos p JOIN clientes c ON p.cliente_id = c.id GROUP BY c.estado;",
     "perguntas": ["Qual o ticket médio dos pedidos por estado?", "Calcule o valor médio dos pedidos de cada estado.",
                   "Mostre a média do valor dos pedidos agrupada por estado.", "Ticket médio por estado do cliente.",
                   "Em cada estado, qual o valor médio de um pedido?"]},
    {"id": "produtos_nunca_vendidos",
     "sql": "SELECT nome FROM produtos WHERE id NOT IN (SELECT produto_id FROM itens_pedido);",
     "perguntas": ["Quais produtos nunca foram vendidos?", "Liste os produtos sem nenhuma venda.",
                   "Mostre os produtos que não aparecem em nenhum pedido.", "Produtos que nunca venderam.",
                   "Existe algum produto que ninguém comprou? Liste-os."]},
    {"id": "pedidos_cidade_status",
     "sql": "SELECT COUNT(*) FROM pedidos p JOIN clientes c ON p.cliente_id = c.id WHERE c.cidade = '{cidade}' AND p.status = '{status}';",
     "perguntas": ["Quantos pedidos de clientes de {cidade} estão {status_pl}?",
                   "Conte os pedidos com status {status} feitos por clientes de {cidade}.",
                   "Número de pedidos {status_pl} de clientes que moram em {cidade}.",
                   "Quantos pedidos {status_pl} vieram de {cidade}?",
                   "Dos clientes de {cidade}, quantos pedidos têm status {status}?"]},
    {"id": "email_cliente",
     "sql": "SELECT email FROM clientes WHERE nome = '{nome}';",
     "perguntas": ["Qual o email do cliente {nome}?", "Me passe o e-mail de {nome}.",
                   "Mostre o endereço de email de {nome}.", "Email de {nome}.",
                   "Como posso contatar {nome} por e-mail?"]},
    {"id": "preco_produto",
     "sql": "SELECT preco FROM produtos WHERE nome = '{produto}';",
     "perguntas": ["Quanto custa o produto {produto}?", "Qual o preço de {produto}?",
                   "Mostre o preço do item {produto}.", "Preço do produto {produto}.",
                   "Por quanto está sendo vendido o {produto}?"]},
    {"id": "media_pedidos_ano",
     "sql": "SELECT AVG(valor_total) FROM pedidos WHERE strftime('%Y', data_pedido) = '{ano}';",
     "perguntas": ["Qual o valor médio dos pedidos em {ano}?", "Calcule a média do valor total dos pedidos de {ano}.",
                   "Ticket médio dos pedidos feitos em {ano}.", "Média de valor por pedido no ano {ano}.",
                   "Em {ano}, quanto valia um pedido em média?"]},
    {"id": "produtos_faixa_preco",
     "sql": "SELECT nome, preco FROM produtos WHERE preco BETWEEN {a} AND {b} ORDER BY preco;",
     "perguntas": ["Quais produtos custam entre {a} e {b} reais?", "Liste os produtos com preço entre {a} e {b}, do mais barato ao mais caro.",
                   "Mostre nome e preço dos produtos na faixa de R$ {a} a R$ {b}.", "Produtos com preço entre {a} e {b}.",
                   "Tenho entre {a} e {b} reais, quais produtos posso comprar? Ordene pelo preço."]},
    {"id": "clientes_distintos_categoria",
     "sql": "SELECT COUNT(DISTINCT p.cliente_id) FROM pedidos p JOIN itens_pedido i ON i.pedido_id = p.id JOIN produtos pr ON pr.id = i.produto_id WHERE pr.categoria = '{cat}';",
     "perguntas": ["Quantos clientes diferentes compraram produtos de {cat}?",
                   "Conte os clientes distintos que compraram algo da categoria {cat}.",
                   "Número de clientes únicos com compras em {cat}.", "Quantos clientes já compraram {cat}?",
                   "Quantas pessoas distintas adquiriram itens da categoria {cat}?"]},
    {"id": "maior_pedido",
     "sql": "SELECT MAX(valor_total) FROM pedidos;",
     "perguntas": ["Qual o maior valor de pedido já registrado?", "Qual foi o pedido mais caro?",
                   "Mostre o valor máximo de um pedido.", "Valor do maior pedido.",
                   "Qual é o valor total mais alto entre todos os pedidos?"]},
    {"id": "ultimos_pedidos",
     "sql": "SELECT id, data_pedido, valor_total FROM pedidos ORDER BY data_pedido DESC LIMIT {n};",
     "perguntas": ["Quais foram os {n} pedidos mais recentes?", "Liste os últimos {n} pedidos.",
                   "Mostre id, data e valor dos {n} pedidos mais novos.", "Os {n} pedidos mais recentes.",
                   "Me traga os {n} últimos pedidos realizados."]},
    {"id": "estoque_por_categoria",
     "sql": "SELECT categoria, SUM(estoque) FROM produtos GROUP BY categoria;",
     "perguntas": ["Qual o estoque total de cada categoria?", "Some o estoque por categoria.",
                   "Mostre a quantidade total em estoque de cada categoria.", "Estoque agrupado por categoria.",
                   "Quantas unidades temos em estoque em cada categoria?"]},
    {"id": "produtos_categoria_preco_maior",
     "sql": "SELECT nome, preco FROM produtos WHERE categoria = '{cat}' AND preco > {valor};",
     "perguntas": ["Quais produtos da categoria {cat} custam mais de {valor} reais?",
                   "Liste nome e preço dos produtos de {cat} com preço acima de {valor}.",
                   "Mostre os produtos {cat} mais caros que R$ {valor}.",
                   "Produtos da categoria {cat} com preço superior a {valor}.",
                   "Preciso dos itens de {cat} que passam de {valor} reais, com seus preços."]},
    {"id": "conta_produtos_categoria",
     "sql": "SELECT COUNT(*) FROM produtos WHERE categoria = '{cat}';",
     "perguntas": ["Quantos produtos existem na categoria {cat}?", "Conte os produtos da categoria {cat}.",
                   "Qual o número de produtos de {cat}?", "Quantos itens de {cat} temos cadastrados?",
                   "Me diga quantos produtos a categoria {cat} possui."]},
    {"id": "conta_clientes_estado",
     "sql": "SELECT COUNT(*) FROM clientes WHERE estado = '{uf}';",
     "perguntas": ["Quantos clientes temos no estado de {estado_nome}?", "Conte os clientes do estado {uf}.",
                   "Qual o número de clientes de {estado_nome}?", "Quantos clientes são do estado {uf}?",
                   "Me diga quantos clientes moram em {estado_nome}."]},
    {"id": "top_clientes_cidade_gasto",
     "sql": "SELECT c.nome, SUM(p.valor_total) AS total FROM clientes c JOIN pedidos p ON p.cliente_id = c.id WHERE c.cidade = '{cidade}' GROUP BY c.id ORDER BY total DESC LIMIT {n};",
     "perguntas": ["Quais são os {n} clientes de {cidade} que mais gastaram?",
                   "Liste os {n} maiores clientes de {cidade} por valor gasto.",
                   "Top {n} clientes de {cidade} por total de compras.",
                   "Mostre os {n} clientes de {cidade} com maior soma de pedidos.",
                   "Em {cidade}, quem são os {n} clientes que mais compraram em valor?"]},
    {"id": "pedidos_mais_caros_ano",
     "sql": "SELECT id, valor_total FROM pedidos WHERE strftime('%Y', data_pedido) = '{ano}' ORDER BY valor_total DESC LIMIT {n};",
     "perguntas": ["Quais foram os {n} pedidos mais caros de {ano}?", "Liste os {n} pedidos de maior valor em {ano}.",
                   "Top {n} pedidos por valor no ano de {ano}.", "Mostre id e valor dos {n} maiores pedidos de {ano}.",
                   "Em {ano}, quais {n} pedidos tiveram o valor total mais alto?"]},
    {"id": "quantidade_vendida_categoria",
     "sql": "SELECT SUM(i.quantidade) FROM itens_pedido i JOIN produtos pr ON i.produto_id = pr.id WHERE pr.categoria = '{cat}';",
     "perguntas": ["Quantas unidades de produtos da categoria {cat} foram vendidas?",
                   "Some a quantidade vendida dos produtos de {cat}.",
                   "Qual o total de itens vendidos da categoria {cat}?",
                   "Quantidade total vendida em {cat}.",
                   "Quantos produtos de {cat} a loja já vendeu, em unidades?"]},
    # ---- templates que NÃO aparecem no treino (tipo inédito) ----
    {"id": "pedidos_estado_status", "inedito": True,
     "sql": "SELECT COUNT(*) FROM pedidos p JOIN clientes c ON p.cliente_id = c.id WHERE c.estado = '{uf}' AND p.status = '{status}';",
     "perguntas": ["Quantos pedidos de clientes do estado {uf} estão {status_pl}?",
                   "Conte os pedidos com status {status} de clientes de {uf}.",
                   "Número de pedidos {status_pl} feitos por clientes do estado {uf}.",
                   "Quantos pedidos {status_pl} vieram do estado {uf}?",
                   "Dos clientes de {uf}, quantos pedidos estão {status_pl}?"]},
    {"id": "preco_medio_estoque_baixo", "inedito": True,
     "sql": "SELECT AVG(preco) FROM produtos WHERE estoque < {estoque};",
     "perguntas": ["Qual o preço médio dos produtos com estoque menor que {estoque}?",
                   "Calcule a média de preço dos produtos com estoque abaixo de {estoque}.",
                   "Preço médio dos itens com menos de {estoque} unidades em estoque.",
                   "Média de preço entre os produtos com estoque inferior a {estoque}.",
                   "Quanto custam em média os produtos que têm menos de {estoque} em estoque?"]},
    {"id": "clientes_cidade_cadastro_ano", "inedito": True,
     "sql": "SELECT nome FROM clientes WHERE cidade = '{cidade}' AND strftime('%Y', data_cadastro) = '{ano}';",
     "perguntas": ["Quais clientes de {cidade} se cadastraram em {ano}?",
                   "Liste os clientes de {cidade} cadastrados no ano de {ano}.",
                   "Mostre os nomes dos clientes de {cidade} com cadastro em {ano}.",
                   "Clientes de {cidade} que entraram em {ano}.",
                   "Quem de {cidade} fez cadastro na loja em {ano}?"]},
]


def _contexto(con: sqlite3.Connection):
    return {
        # só clientes com pedidos, para que consultas por nome não retornem vazio
        "clientes": [r[0] for r in con.execute("SELECT DISTINCT c.nome FROM clientes c JOIN pedidos p ON p.cliente_id = c.id ORDER BY c.nome")],
        "produtos": [r[0] for r in con.execute("SELECT nome FROM produtos")],
    }


def _instanciar(t, idx_pergunta, slots):
    return {
        "template": t["id"],
        "pergunta": t["perguntas"][idx_pergunta].format(**slots),
        "sql": t["sql"].format(**slots),
    }


def _amostrar(rng, t, ctx, idx_perguntas, n, vistas, sqls_proibidos=frozenset(), max_tentativas=5000):
    """Sorteia até n exemplos únicos do template t usando as paráfrases idx_perguntas."""
    saida = []
    for _ in range(max_tentativas):
        if len(saida) >= n:
            break
        ex = _instanciar(t, rng.choice(idx_perguntas), _slots_base(rng, ctx))
        if ex["pergunta"] in vistas or ex["sql"] in sqls_proibidos:
            continue
        vistas.add(ex["pergunta"])
        saida.append(ex)
    return saida


def gerar_dataset(semente: int = 7, n_por_template: int = 60, n_val_por_template: int = 4,
                  n_teste_por_template: int = 10, n_teste_inedito: int = 15, salvar: bool = True):
    """Gera os splits treino/validação/teste e salva em data/*.jsonl."""
    rng = random.Random(semente)
    with conectar(CAMINHO_BANCO) as con:
        ctx = _contexto(con)

    treino, val, teste = [], [], []
    for t in TEMPLATES:
        vistas = set()
        if t.get("inedito"):
            teste += [{**e, "tipo_teste": "tipo_inedito"}
                      for e in _amostrar(rng, t, ctx, range(5), n_teste_inedito, vistas)]
            continue
        exs_treino = _amostrar(rng, t, ctx, [0, 1, 2], n_por_template, vistas)
        treino += exs_treino
        val += _amostrar(rng, t, ctx, [3], n_val_por_template, vistas)
        # teste: metade paráfrase inédita (4), metade paráfrase de treino com valores novos
        teste += [{**e, "tipo_teste": "parafrase_inedita"}
                  for e in _amostrar(rng, t, ctx, [4], n_teste_por_template // 2, vistas)]
        sqls_treino = frozenset(e["sql"] for e in exs_treino)
        teste += [{**e, "tipo_teste": "valores_novos"}
                  for e in _amostrar(rng, t, ctx, [0, 1, 2], n_teste_por_template // 2, vistas, sqls_treino)]

    rng.shuffle(treino)
    rng.shuffle(val)
    if salvar:
        PASTA_DADOS.mkdir(exist_ok=True)
        for nome, dados in [("treino", treino), ("validacao", val), ("teste", teste)]:
            with open(PASTA_DADOS / f"{nome}.jsonl", "w", encoding="utf-8") as f:
                for ex in dados:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return treino, val, teste


def carregar_split(nome: str):
    with open(PASTA_DADOS / f"{nome}.jsonl", encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]

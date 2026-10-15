"""Criação do banco SQLite de exemplo (loja virtual) usado pelo problema Text-to-SQL."""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_BANCO = RAIZ / "data" / "loja.db"

ESQUEMA_SQL = """CREATE TABLE clientes (
  id INTEGER PRIMARY KEY,
  nome TEXT,
  email TEXT,
  cidade TEXT,
  estado TEXT,
  data_cadastro DATE
);
CREATE TABLE produtos (
  id INTEGER PRIMARY KEY,
  nome TEXT,
  categoria TEXT,
  preco REAL,
  estoque INTEGER
);
CREATE TABLE pedidos (
  id INTEGER PRIMARY KEY,
  cliente_id INTEGER REFERENCES clientes(id),
  data_pedido DATE,
  status TEXT,
  valor_total REAL
);
CREATE TABLE itens_pedido (
  id INTEGER PRIMARY KEY,
  pedido_id INTEGER REFERENCES pedidos(id),
  produto_id INTEGER REFERENCES produtos(id),
  quantidade INTEGER,
  preco_unitario REAL
);"""

CIDADES = [
    ("São Paulo", "SP"), ("Campinas", "SP"), ("Santos", "SP"),
    ("Rio de Janeiro", "RJ"), ("Niterói", "RJ"),
    ("Belo Horizonte", "MG"), ("Uberlândia", "MG"),
    ("Curitiba", "PR"), ("Londrina", "PR"),
    ("Porto Alegre", "RS"), ("Florianópolis", "SC"),
    ("Salvador", "BA"), ("Recife", "PE"), ("Fortaleza", "CE"),
    ("Brasília", "DF"), ("Goiânia", "GO"), ("Manaus", "AM"), ("Belém", "PA"),
]

PRODUTOS = {
    "Eletrônicos": [
        ("Smartphone Galaxy A54", 1899.0), ("Notebook Inspiron 15", 3999.0), ("Fone Bluetooth JBL", 299.9),
        ("Smart TV 50 polegadas", 2599.0), ("Mouse Gamer Logitech", 189.9), ("Teclado Mecânico Redragon", 249.9),
        ("Monitor 24 polegadas", 899.0), ("Tablet Galaxy Tab", 1499.0), ("Caixa de Som Echo Dot", 349.0),
        ("Carregador Turbo USB-C", 79.9), ("Smartwatch Amazfit", 599.0), ("Câmera Instax Mini", 499.0),
    ],
    "Livros": [
        ("Dom Casmurro", 29.9), ("O Pequeno Príncipe", 24.9), ("Sapiens", 59.9), ("Hábitos Atômicos", 49.9),
        ("Clean Code", 119.9), ("O Alquimista", 34.9), ("1984", 39.9), ("Grande Sertão: Veredas", 79.9),
        ("Python Fluente", 159.9), ("A Hora da Estrela", 27.9),
    ],
    "Roupas": [
        ("Camiseta Básica Algodão", 49.9), ("Calça Jeans Slim", 159.9), ("Jaqueta Corta-Vento", 229.9),
        ("Vestido Floral", 139.9), ("Moletom Canguru", 179.9), ("Bermuda Sarja", 89.9),
        ("Camisa Social", 129.9), ("Tênis Casual", 249.9),
    ],
    "Casa": [
        ("Jogo de Panelas Tramontina", 449.9), ("Cafeteira Nespresso", 599.0), ("Liquidificador Philips", 229.9),
        ("Aspirador Robô", 1299.0), ("Jogo de Toalhas", 119.9), ("Luminária de Mesa", 89.9),
        ("Air Fryer Mondial", 399.9), ("Edredom Queen", 259.9), ("Ventilador de Coluna", 279.9),
    ],
    "Esportes": [
        ("Bola de Futebol Nike", 149.9), ("Bicicleta Aro 29", 1899.0), ("Halteres 5kg Par", 129.9),
        ("Tapete de Yoga", 99.9), ("Raquete de Tênis Wilson", 499.0), ("Corda de Pular", 39.9),
        ("Garrafa Térmica 1L", 89.9), ("Tênis de Corrida Asics", 549.9),
    ],
    "Brinquedos": [
        ("LEGO Classic", 299.9), ("Boneca Barbie", 129.9), ("Quebra-Cabeça 1000 peças", 69.9),
        ("Carrinho Hot Wheels Pista", 199.9), ("Jogo Banco Imobiliário", 149.9), ("Pelúcia Urso", 79.9),
        ("Cubo Mágico", 39.9),
    ],
}

PRIMEIROS_NOMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Henrique", "Isabela", "João",
    "Larissa", "Lucas", "Mariana", "Mateus", "Natália", "Otávio", "Paula", "Rafael", "Sofia", "Thiago",
    "Vanessa", "Vinícius", "Beatriz", "Gustavo", "Camila", "Rodrigo", "Juliana", "Pedro", "Fernanda", "Leonardo",
]
SOBRENOMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes",
    "Costa", "Ribeiro", "Martins", "Carvalho", "Almeida", "Lopes", "Soares", "Fernandes", "Vieira", "Barbosa",
]

STATUS = ["entregue", "enviado", "pendente", "cancelado"]
PESOS_STATUS = [0.6, 0.15, 0.1, 0.15]


def _data_aleatoria(rng, inicio: date, fim: date) -> date:
    return inicio + timedelta(days=rng.randint(0, (fim - inicio).days))


def criar_banco(caminho: Path = CAMINHO_BANCO, semente: int = 42, n_clientes: int = 200, n_pedidos: int = 1500) -> Path:
    """Cria (ou recria) o banco SQLite com dados sintéticos determinísticos."""
    rng = random.Random(semente)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        caminho.unlink()

    con = sqlite3.connect(caminho)
    con.executescript(ESQUEMA_SQL)

    # clientes com nomes únicos
    nomes = set()
    while len(nomes) < n_clientes:
        nomes.add(f"{rng.choice(PRIMEIROS_NOMES)} {rng.choice(SOBRENOMES)}")
    clientes = []
    for i, nome in enumerate(sorted(nomes), start=1):
        cidade, uf = rng.choice(CIDADES)
        email = nome.lower().replace(" ", ".") + f"{i}@email.com"
        email = email.translate(str.maketrans("áâãéêíóôõúç", "aaaeeiooouc"))
        cad = _data_aleatoria(rng, date(2022, 1, 1), date(2025, 6, 30))
        clientes.append((i, nome, email, cidade, uf, cad.isoformat()))
    con.executemany("INSERT INTO clientes VALUES (?,?,?,?,?,?)", clientes)

    produtos = []
    pid = 1
    for categoria, itens in PRODUTOS.items():
        for nome, preco in itens:
            produtos.append((pid, nome, categoria, preco, rng.randint(0, 120)))
            pid += 1
    con.executemany("INSERT INTO produtos VALUES (?,?,?,?,?)", produtos)

    # alguns produtos nunca vendidos (para tornar consultas interessantes)
    vendaveis = [p for p in produtos if p[0] % 11 != 0]
    # alguns clientes nunca compram
    compradores = [c for c in clientes if c[0] % 13 != 0]

    item_id = 1
    for ped_id in range(1, n_pedidos + 1):
        cli = rng.choice(compradores)
        data_ped = _data_aleatoria(rng, date(2023, 1, 1), date(2025, 9, 30))
        status = rng.choices(STATUS, PESOS_STATUS)[0]
        total = 0.0
        itens = []
        for prod in rng.sample(vendaveis, rng.randint(1, 4)):
            qtd = rng.randint(1, 3)
            itens.append((item_id, ped_id, prod[0], qtd, prod[3]))
            total += qtd * prod[3]
            item_id += 1
        con.execute("INSERT INTO pedidos VALUES (?,?,?,?,?)", (ped_id, cli[0], data_ped.isoformat(), status, round(total, 2)))
        con.executemany("INSERT INTO itens_pedido VALUES (?,?,?,?,?)", itens)

    con.commit()
    con.close()
    return caminho


def conectar(caminho: Path = CAMINHO_BANCO) -> sqlite3.Connection:
    if not Path(caminho).exists():
        criar_banco(caminho)
    return sqlite3.connect(f"file:{caminho}?mode=ro", uri=True)


if __name__ == "__main__":
    print("Banco criado em", criar_banco())

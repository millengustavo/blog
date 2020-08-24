---
title: "cep-to-coords: Pacote Python para converter CEP em coordenadas geográficas (latitude e longitude)"
toc: true
layout: post
categories: [data science, package, python]
image: "/images/cep_to_coords/cep_to_coords.png"
---

Pacote Python para transformar CEP em latitude e longitude

[https://github.com/millengustavo/cep-to-coords](https://github.com/millengustavo/cep-to-coords)

# Como usar:

1. Instale:

```
git clone https://github.com/millengustavo/cep-to-coords.git
cd cep-to-coords
git checkout master
pip install -e .
```

2. Converta um CEP para latitude e longitude:

```
from cep_to_coords.convert import cep_to_coords


coordenadas = cep_to_coords('22070-900')
print(coordenadas)

# {'latitude': -22.9864082, 'longitude': -43.189592}
```

*Opcional - Usando o [https://cepaberto.com/](https://cepaberto.com/)* 

> Para usar esse conversor você precisa se cadastrar no site e exportar seu token como variável de ambiente `CEP_ABERTO_TOKEN`


```
export CEP_ABERTO_TOKEN='seu_token'
```


```
from cep_to_coords.convert import cep_to_coords
from cep_to_coords.strategies import CEPAbertoConverter


coordenadas = cep_to_coords('22070-900', factory=CEPAbertoConverter)
print(coordenadas)

# {'latitude': -22.9864082, 'longitude': -43.189592}
```

# Testando

```
python -m unittest discover
```


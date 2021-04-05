---
title: "IBGE Censo - Associando Localização com Setores Censitários"
toc: true
layout: post
categories: [dados públicos, machine learning, python, data science]
image: "/images/ibge_censo/geospatial_plot.png"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/ibge_censo/geospatial_plot.png" alt="geospatial_plot">

# O Censo do IBGE

> "Em 2010, o IBGE realizou o XII Censo Demográfico, que se constituiu no grande retrato em extensão e profundidade da população brasileira e das suas características sócio-econômicas e, ao mesmo tempo, na base sobre a qual deverá se assentar todo o planejamento público e privado da próxima década. 
> O Censo 2010 é um retrato de corpo inteiro do país com o perfil da população e as características de seus domicílios, ou seja, ele nos diz como somos, onde estamos e como vivemos." - [https://censo2010.ibge.gov.br/sobre-censo.html](https://censo2010.ibge.gov.br/sobre-censo.html)

Para tomar decisões mais assertivas baseadas em dados, dependemos de informações diversas e ricas. Um dado que é comumente requisitado ao usuário para o fornecimento de serviços é o código postal (CEP) ou algum identificador de localização (coordenadas geográficas, endereço, etc). Esse dado deve ser armazenado com precisão reduzida para preservar a privacidade do usuário, mas ainda tem muito valor por levar a associações esclarecedoras, ainda que aproximadas. 

Do Censo de 2010 do IBGE, temos os códigos dos setores censitários, seus polígonos baseados em coordenadas que delimitam regiões brasileiras, e as mais variadas respostas a perguntas sócio-econômicas. Entre essas respostas temos por exemplo a renda, o número de pessoas, o número de banheiros nos domicílios, entre outros. Essas informações são de 2010, portanto alguns ajustes devem ser feitos nas análises. 

# GeoPandas

Para utilizarmos os dados do Censo com dados de localização podemos utilizar uma ferramenta poderosa, o *GeoPandas*. Ele permite que façamos o [Spatial Join](https://geopandas.org/gallery/spatial_joins.html), associando pontos de coordenadas (longitude e latitude) aos seus respectivos polígonos e consequentemente ao seu código de setor censitário. 

Com os códigos para cada localização, as análises posteriores podem ser feitas com um simples join dos agregados usando o código de setor censitário como chave. 

# Dataset no Kaggle 

Disponibilizei no Kaggle um exemplo de aplicação com CEP, coordenadas, código de setor censitário e renda per capita aproximada. 
[https://www.kaggle.com/silveiragustavo/ibge-censo-cep-coordenadas-renda-per-capita](https://www.kaggle.com/silveiragustavo/ibge-censo-cep-coordenadas-renda-per-capita)

O código usado para construir essa base também está aberto no meu Github:
[https://github.com/millengustavo/ibge-censo-cep-coordenadas](https://github.com/millengustavo/ibge-censo-cep-coordenadas)

# Fairness em Machine Learning

É sempre válido lembrar que o analista/cientista que manipula esses dados tem a responsabilidade moral e ética de tratar os indivíduos de forma igualitária ou de forma correta/razoável, o chamado **Fairness**. 

Isso é especialmente importante para algoritmos de Machine Learning cujas decisões podem privar determinados indivíduos do acesso a serviços ou oportunidades. Esse tema é crucial para a construção de um futuro mais justo. Como introdução aos estudos no tema recomendo a talk da Tatyana Zabanova, ["Fairness em Machine Learning Nubank ML Meetup"](https://www.youtube.com/watch?v=LWt4LZmpasc).
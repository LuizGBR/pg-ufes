# Projeto de Graduação -Análise de transferência de aprendizado com rótulos fracos para classificação de câncer de pele
Neste repositório está o código implementado para a realização dos testes usados para análise abordada em meu PG. Forneço algumas instruções sobre como utilizá-lo a seguir.

## Dependências
Todo o código é feito em ambiente Python. Os modelos de Aprendizado de Máquina/Aprendizado Profundo são implementados usando PyTorch.

Para realizar a instalação de todas as dependências, basta executar o comando `pip install -r requirements.txt` (Linux).
## Repositório Raug
Para executar este código, você deve clonar o  [repositório Raug](https://github.com/paaatcha/raug). O Raug é responsável por treinar os modelos de Aprendizado Profundo. Você pode encontrar mais instruções em seu próprio arquivo Readme.md.

Após clonar este repositório, você deve definir o caminho no arquivo constants.py. As instruções estarão lá.

## Organização
Na pasta my_models, você encontrará as implementações dos modelos de CNN. Os mesmos foram aproveitados dos benchmarks disponíveis no [repositório Metablock](https://github.com/paaatcha/MetaBlock/tree/main) Apesar disso, a abordagem descrita em tal repositório não foi utilizada nos experimentos realizados neste trabalho.

Na pasta benchmarks, estão os scripts para os experimentos com os conjuntos de dados REDDIT e PAD-UFES-20.

Para executar os benchmarks, há a opção de se utilizar o Sacred, que é basicamente uma ferramenta para organizar experimentos. 

Não é necessário um conhecimento prévio em Sacred para rodar o código, sendo apenas uma ferramenta de auxilio.

Usando o Sacred, você pode executar um experimento da seguinte maneira, por exemplo: `python reddit.py with _lr=0.001 _batch_size=50`.

Caso escolha por não utilizá-lo, você pode alterar os parâmetros diretamente no código.

Importante: É necessário definir o caminho para o conjunto de dados em benchmark/pad.py e benchmark/reddit.py.

## Onde posso encontrar os conjuntos de dados?
Você pode encontrar os links para todos os conjuntos de dados que utilizei em meu projeto de graduação na seguinte lista:

- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
- [REDDIT IMAGES](https://github.com/Lab-Health/reddit_scraper)

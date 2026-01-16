# Credit Scoring Model

## Visão Geral

Este projeto tem como objetivo o desenvolvimento de um modelo de Credit Scoring para predição de uma variável mascarada y, utilizando técnicas de aprendizado supervisionado e boas práticas amplamente adotadas na indústria financeira.

O trabalho contempla desde a análise exploratória dos dados, engenharia e seleção de variáveis, modelagem estatística, avaliação de performance, até a análise de estabilidade temporal, garantindo robustez, interpretabilidade e reprodutibilidade do modelo.

## Objetivo do Projeto

Desenvolver um modelo capaz de predizer a variável target y, a partir de variáveis preditoras mascaradas, atendendo aos seguintes requisitos:

- Boa capacidade discriminatória<br>
- Estabilidade temporal do modelo<br>
- Interpretabilidade dos resultados<br>
- Código legível, modular e reproduzível<br>
- Clareza na apresentação e documentação<br>

## Descrição da Base de Dados

A base de dados contém 10.738 registros e 81 variáveis, com as seguintes características:

- id: Identificador único da operação<br>
- safra: Mês/ano da concessão do crédito<br>
- y: Variável target<br>
- Demais variáveis: Variáveis preditoras mascaradas<br>

## Observações importantes:

- A variável id não é utilizada na modelagem<br>
- A variável safra é utilizada para validação temporal e análise de estabilidade<br>
- As variáveis preditoras não possuem significado semântico explícito<br>


## Metodologia

A metodologia adotada segue o fluxo tradicional de projetos de Credit Scoring:

1. Análise exploratória dos dados (EDA)<br>
2. Tratamento de dados e valores ausentes<br>
3. Binning das variáveis<br>
4. Transformação WOE (Weight of Evidence)<br>
5. Seleção de variáveis via Information Value (IV)<br>
6. Modelagem supervisionada<br>
7. Validação temporal<br>
8. Avaliação de performance<br>
9. Análise de estabilidade das variáveis e do score<br>
10. Interpretação e apresentação dos resultados<br>


## Modelagem
Algoritmo Principal
Regressão Logística

Justificativa:

- Modelo amplamente utilizado em risco de crédito<br>
- Alta interpretabilidade<br>
- Estabilidade no tempo<br>
- Facilidade de governança e manutenção<br>

Parâmetros:

- max_iter = 2000   | O parâmetro max_iter foi definido como 2000 para garantir a convergência do algoritmo, evitando interrupções prematuras do processo de otimização<br>
- penalty = "l2"    | A regularização L2 foi adotada com o objetivo de reduzir overfitting<br>
- random_state = 42 | O uso de random_state = 42 assegura a reprodutibilidade dos resultados, permitindo que o treinamento do modelo produza os mesmos coeficientes em execuções futuras.<br>


## Métricas de Avaliação

As seguintes métricas foram utilizadas para avaliação do modelo:

- AUC-ROC
- KS (Kolmogorov-Smirnov)

A avaliação foi realizada respeitando a ordem temporal das safras, evitando vazamento de informação (data leakage).


## Estabilidade e Robustez

Para garantir a robustez do modelo, foram realizadas análises de:

- Estabilidade temporal das variáveis<br>
- PSI (Population Stability Index)<br>
- Estabilidade do score ao longo das safras<br>

Essas análises permitem avaliar o comportamento do modelo em diferentes períodos e sua adequação para uso em produção.


## Estrutura do Projeto
credit_scoring/
../data/
../../raw/                     # Base original
../../processed/               # Dados tratados e transformados

../notebooks/
../../01_analise_exploratoria.ipynb   # Primeira análise dos dados
../../02_pre_processamento.ipynb      # Pré-processamento da base de dados
../../03_modelagem.ipynb              # Modelagem e comparação de modelos

../src/
../../pre_processamento.py     # Funções utilizadas no pré-processamento
../../modelagem.py             # Funções utilizadas na modelagem e métricas

../README.md                  # Documentação do projeto
../requirements.txt           # Dependências do projeto


## Reprodutibilidade

Para reproduzir o projeto:

##### Clonar o repositório
git clone <repositorio>

##### Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

##### Instalar dependências
pip install -r requirements.txt

Após a instalação, adicione a base em CSV na pasta data/raw. Então os notebooks podem ser executados na ordem indicada na pasta notebooks/.
## Observações

A execução dos notebooks deve seguir a ordem numérica indicada.
As funções críticas de negócio foram modularizadas na pasta src/, garantindo melhor manutenibilidade e reprodutibilidade.
A separação entre dados brutos e processados evita sobrescrita e facilita auditoria.


## Principais Resultados

- O modelos testados apresentaram boa capacidade discriminatória, separando adequadamente a variavel target.<br>
- As variáveis selecionadas apresentaram Information Value consistente<br>
- O modelo refinado demonstrou estabilidade temporal satisfatória, com PSI dentro dos limites aceitáveis<br>
- A regressão logística mostrou-se adequada para o problema, equilibrando performance e interpretabilidade<br>


## Autor

Henrique Lopes da Silva<br>
Projeto desenvolvido como parte de um desafio técnico de Credit Scoring.

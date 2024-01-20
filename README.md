# ICS5111 Mining Large Scale Data Group Project

You need to design and implement a POC that solves the identified problem and that uses techniques and technologies that were discussed in class. You
can of course, also make use of other techniques and technologies that you think are suitable for the solution.

The POC needs to be packaged in a way that it is easy for us to execute and assess. The use of Jupyter2/Google Colab3 notebooks is encouraged.
Documenting and presenting your work is very important. You are to discuss your contribution as a scientific report that is formatted using the ACM LaTeX class template.

The maximum page limit for this report is of 8 pages (including figures, tables and references).
The title of the report should be related to the selected topic. Furthermore, the report is expected to include the following sections, however you can include further sub-sections if you think that these will improve the structure and readability of the report:
- Introduction: a brief explanation of the problem that is being addressed. This should be concise and highlights the main aim and
objectives (10 marks);
- Related Work: briefly discuss existing research related to the problem addressed. Include literature that uses similar technologies and
technique/s (20 marks);
-  Design and Implementation: this section should include details about the design and implementation of your solution. Consider discussing:
(60 marks):
    - how the data was handled (discuss tasks such as data scraping, data collection, data storage, pre-processing, missing values etc);
    - the architecture/design explaining the how, why, what technologies are used and how your solution leverages open data;
    - the development of the POC to explain and showcase how the solution would work in real life;
    - any experiments performed and how these address objectives;
    - any challenges that were encountered and how they were resolved;
    - Conclusion: provide a critical appraisal of the solution: what were the strong and the weak points the approach used? what worked well and what did not and how this work can be extended in the future (10marks);
- Deliverable will be marked out of 100%, however it is equivalent to 80% of the total mark

## Contributions

- Adam
  - Discovery & proposal of benchmark paper.
  - Bootstrapping, refactoring and re-engineering of paper's models and code. See [github](https://github.com/adamd1985/socialmedia_ai_analysis_mentalhealth_predictiveintervention).
  - Preprocessing and large text data sanitization pipelines.
  - Senitiment and lexicon analysis and [research](./sentiment_resaerch.ipynb).
  - Feature engineering. See [post feature engineering](./post_feature_engineering.ipynb).
  - Graph Neural Network. See [GNN](./gnn.ipynb).
- Christian
  - Medical papers research and selection.
  - Bootstrapping of benchmark paper data.
  - Semantic Relation Ship Analysis and [research](./SemanticRelationship.ipynb).
  - Building of graph network (./GraphBuilding.ipynb).
- Owen
  - Github administration.
  - Bag of Words Analysis (./bag_of_words.ipynb).

# MHA Application

## Install and Commit

Create *python 3.10* env:
`conda create -n aml python=3.10 & conda activate bd_mha`

Install dependencies from **requirements.txt**:
`yes | pip install -r requirements.txt`

[Install](https://spacy.io/usage) SpacY English Models:
`python -m spacy download en_core_web_sm`

## Commiting

Use pipreqs to help commit dependencies:
- `pip install pipreqs`
- `pip install nbconvert`

When **commiting**, update the requirements from your base folder:
`jupyter nbconvert --to script *.ipynb & pipreqs . --force`

**NB**: the above will generate a file **.py*, this is only needed for pipreqs to capture dependecies. Do not commit (all python files go to the .ignore)

## Datasets

Commit datasets as CSV if <10mb, else use LSF.

All CVSs should go to the folder ['./raw_data'](./raw_data)

## ENV configurations

A default `.env` was provided.
Use your own and add it to `.gitignore`.
For the reddit scraper have these in your env:
```yml
CLIENT_ID=YOUR_CLIENT_ID
CLIENT_SECRET=YOUR_SECRET
USER_AGENT=uom_research v1.0 by /u/YOUR_USER_NAME
SUBREDDITS=MentalHealth,depression,anxiety
```
The authentication  key and secrets you need to create from this wiki: https://www.reddit.com/wiki/api/

API connections, environment variables, and other secrets need to go in your .env.
**Do not commit** your .env!
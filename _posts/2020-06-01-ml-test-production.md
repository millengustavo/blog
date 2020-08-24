---
title: "Machine Learning: tests and production"
toc: true
layout: post
categories: [tests, machine learning, data science, deploy]
image: "/images/ml_test_production/systems_comparison.PNG"
---

> "Creating reliable, production-level machine learning systems brings on a host of concerns not found in small toy examples or even large offline research experiments. Testing and monitoring are key considerations for ensuring the production-readiness of an ML system, and for reducing technical debt of ML systems." - [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/pub46555/)

I recently read the excellent book written by Emmanuel Ameisen: [Building Machine Learning Powered Applications Going from Idea to Product](http://shop.oreilly.com/product/0636920215912.do)

I definitely recommend the book to people involved at any stage in the process of developing and implementing products that use Machine Learning.

# The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction
A good reference I found in chapter 6 of the book entitled: Debug your ML problems, was the article written by Google engineers: [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/pub46555/)

In the article, the authors:
> "(...) present 28 specific tests and monitoring needs, drawn from experience with a wide range of production ML systems to help quantify these issues and present an easy to follow road-map to improve production readiness and pay down ML technical debt."

As a good practice presented by Ameisen in his book when referring to reproducing successful results from the past: **"Stand on the shoulders of giants"**, I believe that we can learn from Google's experience in building applications using Machine Learning.

## Manually coded systems vs. ML-based systems
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/systems_comparison.PNG" alt="systems_comparison">

Unlike manually coded systems, the behavior of machine learning systems depends on data and models that are not always possible to specify fully a priori

> "(...) training data needs testing like code, and a trained ML model needs production practices like a binary does, such as debuggability, rollbacks and monitoring"

## Tests
### Data
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/data.PNG" alt="data">

### Model
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/model.PNG" alt="model">

### Infrastructure
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/infra.PNG" alt="infra">

### Monitoring
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/monitor.PNG" alt="monitor">

## Computing an ML Test Score
<img src="{{ site.url }}{{ site.baseurl }}/images/ml_test_production/score.PNG" alt="score">

## Insights from applying the rubric to real systems
- **Checklists** are helpful even for expert teams
- **Data dependencies** can lead to outsourcing responsibility for fully understanding it
- The importance of **frameworks**: pipeline platforms may allow building generic integration tests

# Conclusion
Deploying a machine learning system with monitoring is a very complex task. This is a problem faced by virtually all players in the market who are starting their journey with data.

A good first step on this journey is to organize your data pipeline and use managed environments in the cloud for the ML tasks:
- Amazon SageMaker
- Google Cloud AI Platform
- Azure Machine Learning Studio
- Databricks

Even if we do not face the scale of some problems mentioned in the article, it is worth reflecting on how we can improve what we do today to reduce technical debt in the future.


# Further reading
- [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/pub46555/)
- [Building Machine Learning Powered Applications Going from Idea to Product - Emmanuel Ameisen](http://shop.oreilly.com/product/0636920215912.do)
- [ML in Production blog](https://mlinproduction.com/)
- [What is your machine learning test score?](https://speakerdeck.com/trallard/what-is-your-ml-test-score)
- [MLflow: A Machine Learning Lifecycle Platform](https://github.com/mlflow/mlflow)
- [Databricks for Machine Learning](https://databricks.com/solutions/machine-learning)

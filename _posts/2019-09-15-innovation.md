---
title: "Innovation and Operations"
toc: true
layout: post
categories: [sre, innovation, devops, development, operations]
image: "/images/innovation/slide1.jpeg"
---

I was at Google's meetup a few weeks ago about DevOps and SRE (*Site Reliability Engineering*).

<img src="{{ site.url }}{{ site.baseurl }}/images/innovation/selfie.jpeg" alt="meetup">

While these themes are not part of the essence of Data Science such as statistics, math, and programming, it is possible to draw relevant insights to technology in general.

In the term **DevOps**, there are morphologically two sectors:
- **Developers** looking to implement new features quickly, which can mean innovation.
- **Operations**, which ensure customers will be served consistently.

While some (dev) want to create new features as quick as possible, others (ops) like what is there working, and have an aversion to uncertainty.

<img src="{{ site.url }}{{ site.baseurl }}/images/innovation/devops.jpeg" alt="devops">

Both views are extremely valid, and neither is more correct than the other. It is like a balance. For sectors to generate value consistently, the balance should not weigh heavily on either side.

# Error Budget

Most technology service provision agreements have an SLA. An acceptable maximum delay for delivery of the service or product is set.

<img src="{{ site.url }}{{ site.baseurl }}/images/innovation/glossario.jpeg" alt="glossario">

This concept, in addition to guiding the quality of operations, provides an interesting insight into freedom of innovation.

Imagine that the service is being delivered at a consistency far above that agreed upon. For operations this is the perfect scenario, nothing fails, we have no errors.

But this view is not 100% correct if you think about business in the medium and long term. If there is room to make more mistakes, the business must be free to innovate. This is where developers come in and can spend their error budget on new products or enhancements. 

<img src="{{ site.url }}{{ site.baseurl }}/images/innovation/sre.jpeg" alt="sre">

The **error budget** provides a metric that determines how unreliable the service is allowed to be. This metric removes the politics from negotiations when deciding how much risk to allow.

The main benefit of an error budget is that it provides a common incentive that allows both product development and SRE to focus on finding the right balance between innovation and reliability.

# Key Insights from Google

- Managing service reliability is largely about managing risk, and managing risk can be costly.
- 100% is probably never the right reliability target: not only is it impossible to achieve, it’s typically more reliability than a service’s users want or notice. Match the profile of the service to the risk the business is willing to take.
- An error budget aligns incentives and emphasizes joint ownership between SRE and product development. Error budgets make it easier to decide the rate of releases and to effectively defuse discussions about outages with stakeholders, and allows multiple teams to reach the same conclusion about production risk without rancor.

## References
- SRE Book by Google - https://landing.google.com/sre/sre-book/chapters/embracing-risk/
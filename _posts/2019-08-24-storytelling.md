---
title: "Storytelling with Data"
toc: true
layout: post
categories: [data science, storytelling, data driven, book]
image: "/images/storytelling/storytelling.jpg"
---

Gustavo is a data scientist that is trying to grow a **data driven** culture in the company he works for.

Many hours of study of scientific method, statistics and programming have made him technically competent in his role.

For him, analyzing charts coming from his usual tools is trivial. However, Gustavo works with other amazing people in different roles from his. His goal of making the company data driven depends on their **engagement**.

To fulfill his goal, our hero embarks on an arduous journey to make his visualizations more assertive and explain his points simply and elegantly.

Make no mistake, often summarizing an idea to the essentials is a much more complex task than expected.

The above seems like the synopsis of a book, doesn't it? That was the way I found it most interesting to engage the reader inspired by the chapter *“The magic of story”* from the book Storytelling with Data.

This post is my summary of interesting points from the book. As a three-act structure for plays, we had the **setup** and **conflict** already laid down. The following is my attempt of **resolution**. 

# Exploratory vs Explanatory

After undertaking an entire analysis, it can be tempting to want to show your audience everything, as evidence of all of the work you did and the robustness of the analysis (**Exploratory**). Resist this urge. Concentrate on the information your audience needs to know (**Explanatory**).

Being concise is often more challenging than being verbose. If you know exactly what it is you want to communicate, you can make it fit the time slot you’re given. 

# Understand the context

Build a clear understanding of who you are communicating to, what you need them to know or do, how you will communicate to them, and what data you have to back up your case. Employ concepts like the 3‐minute story, the Big Idea, and storyboarding to articulate your story and plan the desired content and flow.

## The Big Idea

1. Must articulate your unique point of view;
2. Must convey what’s at stake;
3. Must be a complete sentence.

# Choosing an effective visual

<img src="{{ site.url }}{{ site.baseurl }}/images/storytelling/visual.jpg" alt="visual">

- **Simple text**: When you have a number or two that you want to communicate, think about using the numbers themselves. When you have more data that you want to show, generally a table or graph is the way to go.
- **Tables** and **Graphs**: While tables interact with our verbal system, graphs interact with our visual system, which is faster at processing information.
- **Scatterplots** can be useful for showing the relationship between two things.
- **Line graphs** are most commonly used to plot continuous data.
- **Slopegraphs** can be useful when you have two time periods or points of comparison and want to quickly show relative increases and decreases or differences across various categories between the two data points.
- **Bar charts** are easy for our eyes to read. Our eyes compare the endpoints of the bars, so it is easy to see quickly which category is the biggest, which is the smallest, and also the incremental difference between categories. Note that, because of how our eyes compare the relative end points of the bars, it is important that bar charts always have a zero baseline (where the x‐axis crosses the y‐axis at zero), otherwise you get a false visual comparison.
- **Stacked vertical bar chart** are meant to allow you to compare totals across categories and also see the subcomponent pieces within a given category.
- **Horizontal bar chart** are the go-to graph for categorical data. They are especially useful if your category names are long, as the text is written from left to right, making your graph legible for your audience.
- **Avoid**: pie charts, donut charts, 3D and secondary y-axes.

# Focus your audience’s attention

- If we use **preattentive attributes** strategically, they can help us enable our audience to see what we want them to see before they even know they’re seeing it!
- Don’t let your design choices be happenstance; rather, they should be the result of **explicit decisions**.
- The use of **color** should always be an intentional decision. Never let your tool make this important decision for you!
- If something is important, try not to make your audience wade through other stuff to get to it. Eliminate this work by putting the important thing at the top.

# Think like a designer

> Antoine de Saint‐Exupery famously said, “You know you’ve achieved perfection, not when you have nothing more to add, but when you have **nothing to take away**”

- Ask yourself: would eliminating this change anything? No? Take it out! Resist the temptation to keep things because they are cute or because you worked hard to create them; if they don’t support the message, they don’t serve the purpose of communication.
- Push necessary, but non‐message‐impacting items to the background. Use your knowledge of preattentive attributes to deemphasize. Light grey works well for this.
- Make it legible; Keep it clean; Use straightforward language; Remove unnecessary complexity.
- Assume that every chart needs a title and every axis needs a title (exceptions to this rule will be extremely rare). The absence of these titles—no matter how clear you think it may be from context—causes your audience to stop and question what they are looking at. Instead, **label explicitly** so they can use their brainpower to understand the information, rather than spend it trying to figure out how to read the visual.
- In data visualization—and communicating with data in general—spending time to make our designs **aesthetically pleasing** can mean our audience will have more patience with our visuals, increasing our chance of success for getting our message across.

# Lessons in Storytelling

- A good story grabs your attention and takes you on a journey, evoking an **emotional response**. In the middle of it, you find yourself not wanting to turn away or put it down. After finishing it—a day, a week, or even a month later—you could easily describe it to a friend.
- Aristotle introduced a basic but profound idea: that story has a clear beginning, middle, and end. He proposed a **three‐act structure for plays**. This concept has been refined over time and is commonly referred to as the setup, conflict, and resolution.

> What exactly is story? At a fundamental level, a story expresses how and why life changes. Stories start with balance. Then something happens—an event that throws things out of balance. McKee describes this as “subjective expectation meets cruel reality.”

- Telling a compelling story is harder than conventional rhetoric. But delving into your creative recesses is worth it because a story allows you to engage your audience on an entirely new level.
- **End with a call to action**: make it totally clear to your audience what you want them to do with the new understanding or knowledge that you’ve imparted to them. One classic way to end a story is to tie it back to the beginning

In his book, Beyond Bullet Points, Cliff Atkinson outlines the following questions to consider and address when it comes to setting up the story:

1. The setting: When and where does the story take place?
2. The main character: Who is driving the action? (This should be framed in terms of your audience!)
3. The imbalance: Why is it necessary, what has changed?
4. The balance: What do you want to see happen?
5. The solution: How will you bring about the changes?

# Final thoughts

The default settings of our graphing application are typically far from ideal. **Our tools do not know the story we aim to tell**. Combine these two things and you run the risk of losing a great deal of potential value—including the opportunity to drive action and effect change—if adequate time isn’t spent on this final step in the analytical process: the communication step.

As a Python programmer I found this amazing Github repository with code to plot the same graphs the author presents in her book using the library Matplotlib:
https://github.com/empathy87/storytelling-with-data

I intentionally did not present any graphs here to encourage you to read the book and navigate through the examples that are beautifully laid out by the author.

> “Nirvana in communicating with data is reached when the effective visuals are combined with a powerful narrative.”

### Reference:

Knaflic, Cole Nussbaumer. Storytelling with data: A data visualization guide for business professionals. John Wiley & Sons, 2015.

Header: Photo by [Nong Vang](https://unsplash.com/@californong?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/storytelling?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

Body: Photo by [Benjamin Smith](https://unsplash.com/@ifbdesign?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)


# Hiver Support Agent Report

## Problem Framing

The objective of this project is to build an automated support agent capable of classifying, handling, and escalating inbound customer support queries on Twitter. We selected **AmazonHelp** as the target brand due to its high inbound volume and the recognizable, repeatable nature of its support requests (e.g., delivery delays, missing packages, account issues).

### Exploratory Data Analysis (EDA)
A quantitative pass over our subsample of 30,000 interactions reveals that customer messages are concise (averaging 117 characters), while brand replies are similarly brief (averaging 124 characters). 

Qualitative analysis of 100 random threads highlights that the majority of interactions revolve around delivery logistics, payment/account access, and missing packages. Because Amazon handles sensitive order information, the typical brand resolution on Twitter involves apologizing, asking for a tracking update, or directing the user to a secure authentication link/chat to discuss account details. Notably, the dataset is highly multilingual and frequently features frustrated or sarcastic tones. We also observe the brand actively warning customers who inadvertently post personal information (like order numbers) publicly.

These insights inform our taxonomy and evaluation strategy. "Good" handling for AmazonHelp doesn't necessarily mean resolving the issue fully in the tweet; it often means correctly identifying the issue type, displaying empathy, and providing the correct next step or link, while escalating complex or high-frustration cases to human agents.

*(More sections will be added as we progress through the build plan)*

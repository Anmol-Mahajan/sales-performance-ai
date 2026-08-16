# LinkedIn post

I have been building a sales performance application that currently runs entirely on my local machine using MSP sales data.



It brings together revenue, pipeline, meetings, customer health, contracts, renewals, service opportunities, escalations and delivery projects. A manager can see where performance is changing, which customers need attention and which actions could protect or grow revenue.



The local question engine turns requests into checked pandas queries. It supports salesperson, customer and flexible date filters, then shows the interpretation and source with the answer. I chose the 2.5 GB `qwen3:4b-instruct` model through Ollama because it is practical on my local machine with 16 GB RAM. It explains verified results locally, it does not calculate the figures or make the revenue prediction.



The Model Analysis view compares four algorithms using a six-month temporal holdout. New feature groups are re-evaluated from the start instead of being assumed to improve the model. In this test, meeting and opportunity note features made accuracy worse, so they were not promoted.



The next step towards production would be to replace the local workbook with an actual CRM and operational data, add access control, schedule validation and retraining, and monitor data quality and model performance. 



In production, this could reduce manual reporting and help teams focus earlier on pipeline gaps, renewals, cross sell opportunities, unanswered customer requests and delivery risks. The current results remain a local demonstration and results can change with live data.

#MachineLearning #LocalAI #DataScience #SalesAnalytics

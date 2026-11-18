# Presentation: speaker notes

The script for the 5-minute [presentation](presentation.pdf), one block per slide. It is about 650 words, roughly 4½ to 5 minutes at a calm pace. The slide source is [slides/presentation.html](slides/presentation.html).

## 1. How long will this taxi trip take? (15 s)

Hi, I'm Sean. My project takes raw New York taxi data all the way to a deployed machine learning service. You send it a trip, and it tells you how many minutes the trip will take.

## 2. The brief, and what I added (35 s)

The brief: take the January 2025 Yellow Taxi trips, about three and a half million rides, train a Random Forest, track the experiments in MLflow, and serve predictions with FastAPI. Bonus points for tests and tuning. I did all of that. I also made sure the model only uses what is known when a trip is booked, validated on later days instead of a random sample, and wrote down how I'd pick the winner before I tuned anything.

## 3. Five steps from a raw file to a live API (40 s)

Here's how it works. One command downloads the data. The cleaning step keeps trips between 1 and 60 minutes and splits the month: the first three weeks to train, the last ten days to test. Then I compare models: four baselines and twenty Optuna tuning trials, all tracked in MLflow. A rule picks the winner, and it gets promoted in the MLflow model registry. Finally, Docker Compose starts the API, which loads whichever model is marked production. Training and the API share one feature function, so they can't drift apart.

## 4. Rush hour halves the speed (30 s)

Before modelling, I explored the data. Almost all trips last 1 to 60 minutes; the rest are errors like negative durations. Fare and tip are only known after the trip, so they're never features. The chart shows why time matters: at 5 in the morning taxis do 16 miles an hour, during the day only 8 or 9. So pickup hour and weekday became features.

## 5. 24 runs, one rule picks the winner (40 s)

This is the comparison from MLflow. Guessing the average is off by almost 10 minutes. The course's linear model gets to about 5. A forest alone is no better; adding the time features brings it to 4.7. Tuning got to 4.61, but that model is 575 megabytes. My rule, set before tuning: within 3 seconds of the best, take the smallest. So I serve trial 16: half a second less accurate, and 3.6 times smaller.

## 6. The distance column knew the answer (45 s)

The most important thing I learned. My first model used the trip distance column, like the course solution. It scored 4.02. But that is the distance actually driven, and you only know it after the trip. A booking app can't send it. So I replaced it with the typical distance between the two zones, learned from the training trips. The honest score is 4.62, about 0.6 minutes worse. Version 1 is still in the registry, marked retired, with the reason.

## 7. Ask it about a trip (35 s)

Here's the service. I send pickup zone, dropoff zone and pickup time, and get back minutes plus the model version that answered. Midtown to the Upper East Side on a Thursday at half past five: 14 minutes. The same trip early on Sunday: 8. To JFK at rush hour: 54. Behind it are 67 tests in CI, 9 behaviour tests on the real model, and a Docker check on every pull request.

## 8. Reliable for everyday trips, weak at the extremes (35 s)

How good is it? On days it never saw, the typical miss is 2.3 minutes, and 82% of trips are within 5 minutes. Where it struggles: it pulls extreme trips to the middle, so hour-long trips come out about 12 minutes short. Next, I'd use a routing estimate per trip, try gradient boosting, and watch for drift: the last week of January was already 6% slower. Thank you, I'm happy to take questions.

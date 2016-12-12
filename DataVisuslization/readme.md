Titanic Data Analysis

Summary:

This project tries to find the survival patterns,if any, of the passengers in the Titanic accident. We analyze the percentage of Survivors with respect to different parameters 
such as gender,age,Pclass,etc. The data visualization created as part of this project is an attempt to convey our findings about the data to a broader audience in an effective 
manner.

More information about the data is present in the link below:
https://www.kaggle.com/c/titanic/data

Design:
At first, I thought a line chart would be a good choice for showcasing the trend in the data. The Percentage of Survivors is the primary measure which the audience needs to see from 
different perspectives such as Age, Passenger Class and Gender. Hence, Position y was selected for Percentage of Survivors. Gender was selected as Legend.

At first, the age was split into buckets of 15. But, I wanted to highlight the differences in survival pattern between Children and Adult. Hence, I regrouped the passengers into two
groups Children and Adults. However, on slicing by Agegroup, the survival pattern among passenger classes gets lost. Hence, I created a third group "All", to focus on Passenger Class only.

As per the feedback, I tried using bar chart and it turned out be a much better option for representing categorical data.

Feedback:
Link:
https://discussions.udacity.com/t/need-feedback-for-visualization-please-help/201295/2

Feedback 1:
Hi Kishan,

Good work! I have two quick suggestons:

Consider label the axes for easier understanding. Also consider adding some narrative to highlight the message you want to convey, then you can optimize your visualization based on that;
The line chart does show the main structure in the data, but I wonder whether a bar chart like1 this can make the viz even more visually engaging and pleasing.
I hope this helps.

Feedback 2:

What do you notice in the visualization? 
	More females survived
What questions do you have about the data? 
	Is the data authentic?
What relationships do you notice? 
	Male survival rate tends to decrease with age and Pclass
What do you think is the main takeaway from this visualization? 
	Females tend to survive more 
Is there something you don’t understand in the graphic? 
	no

Feedback 3:

Please add labels for better clarity.

Resources:
https://discussions.udacity.com/t/need-feedback-for-visualization-please-help/201295/2
http://stackoverflow.com/questions/17275662/repeating-transitions-in-d3-using-setinterval
https://bost.ocks.org/mike/transition/
http://dimplejs.org/advanced_examples_viewer.html?id=advanced_storyboard_control

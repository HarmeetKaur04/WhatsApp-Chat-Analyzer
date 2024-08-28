import urlextract
extractor = urlextract.URLExtract()
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
from transformers import pipeline
def fetch_stats(selected_user,df):

    if selected_user!="Overall":
        df = df[df['users'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())


    # fetch media messages
    num_media_messages=df[df['message']=='<Media omitted>\n'].shape[0]

    # fetch number of links shared


    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    return num_messages, len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['users'].value_counts()
    df=round((df['users'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percent'})
    return x,df
def create_wordcloud(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df=df[df['users']==selected_user]

    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'Missed video call\n']
    temp = temp[temp['message'] != 'Missed voice call\n']
    def remove_stop_word(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc=WordCloud(width=500,height=500,background_color='white',max_font_size=100,min_font_size=50)
    temp['message']=temp['message'].apply(remove_stop_word)
    df_wc=wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc
def most_common_word(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['users']==selected_user]
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'Missed video call\n']
    temp = temp[temp['message'] != 'Missed voice call\n']

    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df[1:]
def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['users']==selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df
def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['users']==selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time']=time
    return timeline
def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline
def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['day_name'].value_counts()
def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
    
def analyze_sentiments(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['users']==selected_user]
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'Missed video call\n']
    temp = temp[temp['message'] != 'Missed voice call\n']

    messages = temp['message'].tolist()

    sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    sentiment_results = sentiment_analyzer(messages)
    return list(zip(messages, sentiment_results))





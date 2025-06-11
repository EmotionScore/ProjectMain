# Install required packages if not already installed
install.packages(c("tm", "wordcloud", "wordcloud2", "textTinyR", "ggplot2", "reticulate", "Matrix"))

# Load libraries
library(Matrix)
library(tm)
library(wordcloud)
library(wordcloud2)
library(textTinyR)
library(ggplot2)
library(reticulate)

# Set working directory and paths
wd <- "/Users/ulviyaabasova/Desktop/replication package"
wd_aux <- paste0(wd, "/data/3_auxiliary_data")

# Optional: Print path and files to verify
print(wd_aux)
print(list.files(wd_aux))

# --- Load Python pickle file ---
pickle <- import("pickle")
builtins <- import_builtins()

# Load the pickle file
file_path <- paste0(wd_aux, "/word_freqs.pkl")
file <- builtins$open(file_path, "rb")
freqs <- pickle$load(file)
file$close()

# Convert to R list
freqs_r <- py_to_r(freqs)

# --- Convert R list to DataFrame ---
freqs_df <- data.frame(
  word = names(freqs_r),
  freq = unlist(freqs_r),
  stringsAsFactors = FALSE
)

# --- Clean and Filter Data ---
# Remove NAs and non-positive frequencies
freqs_df <- freqs_df[!is.na(freqs_df$freq) & freqs_df$freq > 0, ]

# Keep only reasonably frequent words
freqs_df_clean <- freqs_df[freqs_df$freq > 0.01, ]

# Remove words with digits or fewer than 3 letters
freqs_df_clean <- freqs_df_clean[grepl("^[a-z]{3,}$", freqs_df_clean$word), ]

# --- Preview Top Words ---
head(freqs_df_clean[order(-freqs_df_clean$freq), ])

# --- Generate Word Cloud ---
wordcloud(
  words = freqs_df_clean$word,
  freq = freqs_df_clean$freq,
  max.words = 100,
  min.freq = 2,
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2"),
  scale = c(4, 0.5)
)

library(ggplot2)

# Top 20 words by frequency
top_words <- freqs_df_clean[order(-freqs_df_clean$freq), ][1:20, ]

# Bar chart
ggplot(top_words, aes(x = reorder(word, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 20 Frequent Words",
    x = "Word",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 14)

ggplot(freqs_df_clean, aes(x = freq)) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(
    title = "Word Frequency Distribution",
    x = "Frequency",
    y = "Count of Words"
  ) +
  theme_minimal(base_size = 14)







freqs_df_clean$rank <- rank(-freqs_df_clean$freq)

ggplot(freqs_df_clean, aes(x = log(rank), y = log(freq))) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  labs(
    title = "Log-Log Plot: Word Rank vs. Frequency",
    x = "Log(Rank)",
    y = "Log(Frequency)"
  ) +
  theme_minimal(base_size = 14)


library(wordcloud2)

wordcloud2(data = freqs_df_clean, size = 0.7, color = "random-dark", shape = "circle")
install.packages("tidytext")
library(tidytext)
library(dplyr)

# Example if you have multiple documents
# Assuming you have a dataframe: doc_df with columns `doc_id` and `text`
library(tidytext)
library(dplyr)
library(ggplot2)

# Join with sentiment dictionary (e.g. Bing lexicon)
sentiment_scores <- freqs_df_clean %>%
  inner_join(get_sentiments("bing"), by = c("word"))

# Summarize by sentiment
sentiment_summary <- sentiment_scores %>%
  group_by(sentiment) %>%
  summarise(total_freq = sum(freq)) %>%
  arrange(desc(total_freq))

# Plot
ggplot(sentiment_summary, aes(x = sentiment, y = total_freq, fill = sentiment)) +
  geom_bar(stat = "identity") +
  labs(title = "Sentiment Composition of Most Frequent Words", y = "Total Frequency") +
  theme_minimal()


#install.packages("textstem")




freqs_df_clean$rank <- rank(-freqs_df_clean$freq)

ggplot(freqs_df_clean, aes(x = log(rank), y = log(freq))) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "black") +
  labs(title = "Zipf's Law: Word Frequency vs. Rank (Log Scale)",
       x = "Log(Rank)", y = "Log(Frequency)") +
  theme_minimal()










list.files("/Users/ulviyaabasova/Desktop/replication package/data/dictionaries", full.names = TRUE)


# Simple thematic tagging
freqs_df_clean$theme <- ifelse(freqs_df_clean$word %in% c("climate", "energy", "environment"), "Environment",
                               ifelse(freqs_df_clean$word %in% c("tax", "economy", "job"), "Economy",
                                      ifelse(freqs_df_clean$word %in% c("fear", "hope", "anger"), "Emotion", "Other")))

table(freqs_df_clean$theme)

# Visualize
ggplot(freqs_df_clean, aes(x = theme, y = freq, fill = theme)) +
  geom_col(stat = "identity") +
  theme_minimal() +
  labs(title = "Word Themes in Frequency Data")



# Install required packages (if not already installed)
install.packages(c("tm", "wordcloud", "wordcloud2", "textTinyR", "text", "ggplot2", "reticulate"))
install.packages("Matrix")
# Load libraries
library(Matrix)
library(tm)
library(wordcloud)
library(wordcloud2)
library(textTinyR)
library(ggplot2)
library(reticulate)

# Set working directory
wd <- "/Users/ulviyaabasova/Desktop/xxx"

# Set paths
wd_data <- paste0(wd, "/data/1_main_datasets")
wd_results <- paste0(wd, "/results/main_paper")
wd_aux <- paste0(wd, "/data/3_auxiliary_data")
wd_model <- paste0(wd, "/models")

# Optional: print path and files to verify
print(wd_aux)
print(list.files(wd_aux))

# --- Load Python pickle file ---
# Import Python modules
pickle <- import("pickle")
builtins <- import_builtins()

# Load the pickle file
file_path <- paste0(wd_aux, "/word_freqs.pkl")
file <- builtins$open(file_path, "rb")
freqs <- pickle$load(file)
file$close()

# Convert to R-compatible format
freqs_r <- py_to_r(freqs)

# Check structure
str(freqs_r)

# --- Example: Convert to data frame if needed ---
# If it's a named list or dictionary-like structure
if (is.list(freqs_r)) {
  freqs_df <- data.frame(word = names(freqs_r), freq = unlist(freqs_r), stringsAsFactors = FALSE)
} else if (is.data.frame(freqs_r)) {
  freqs_df <- freqs_r
} else {
  stop("Unexpected data format after conversion")
}
# Initialize empty vectors
words <- character()
freqs <- numeric()

# Get Python's built-in list function
py_list <- import_builtins()$list

# Convert dict_items to a proper Python list of tuples, then convert to R
items <- py_to_r(py_list(freqs_r$items()))

# Pre-allocate vectors (faster than appending in a loop)
n <- length(items)
words <- character(n)
freqs <- numeric(n)

# Extract keys and values safely
for (i in seq_len(n)) {
  pair <- items[[i]]
  words[i] <- as.character(pair[[1]])
  freqs[i] <- as.numeric(pair[[2]])
}

# Combine into a data frame
freqs_df <- data.frame(
  word = words,
  freq = freqs,
  stringsAsFactors = FALSE
)

# Preview result
str(freqs_df)
head(freqs_df[order(-freqs_df$freq), ])



# Keep only reasonably frequent words
freqs_df_clean <- freqs_df[freqs_df$freq > 0.01, ]

# Remove words with digits or fewer than 3 letters
freqs_df_clean <- freqs_df_clean[grepl("^[a-z]{3,}$", freqs_df_clean$word), ]

wordcloud(words = freqs_df_clean$word, freq = freqs_df_clean$freq,
          max.words = 100, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"), scale = c(4, 0.5))






# --- Optional: Preview top words ---
head(freqs_df[order(-freqs_df$freq), ])

# --- Example: Basic Word Cloud ---
wordcloud(words = freqs_df$word, freq = freqs_df$freq, min.freq = 2,
          max.words = 100, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))


warnings()

# Remove NAs and zero or negative frequencies
freqs_df_clean <- freqs_df[!is.na(freqs_df$freq) & freqs_df$freq > 0, ]

wordcloud(
  words = freqs_df_clean$word,
  freq = freqs_df_clean$freq,
  min.freq = 2,
  max.words = 100,
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2")
)


list.files("/Users/ulviyaabasova/Desktop/xxx/data/1_main_datasets", full.names = TRUE)
library(readr)
doc_df <- read_csv("/Users/ulviyaabasova/Desktop/xxx/data/1_main_datasets/main_dataset.csv")

# View structure and first rows
str(doc_df)
head(doc_df)
head(doc_df, 2)
colnames(doc_df)

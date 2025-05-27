# Install required libraries if not already installed
install.packages(c("ggplot2", "dplyr", "readr"))

# Load libraries
library(ggplot2)
library(dplyr)
library(readr)


# Set working directory
wd <- "/Users/ulviyaabasova/Desktop/xxx"  # Replace with your actual directory

# Set data and results paths
wd_data <- paste0(wd, "/data/1_main_datasets")
wd_results <- paste0(wd, "/results/main_paper")

# Load dataset
df <- read_csv(paste0(wd_data, "/main_dataset.csv"))
# Calculate mean score and standard error
d1 <- df %>%
  group_by(chamber, speech_year) %>%
  summarize(mean_score = mean(score, na.rm = TRUE))

d2 <- df %>%
  group_by(chamber, speech_year) %>%
  summarize(std_error = sd(score, na.rm = TRUE) / sqrt(n()))

# Merge mean and standard error
final <- left_join(d1, d2, by = c("chamber", "speech_year"))

# Convert speech_year to numeric and sort
final$speech_year <- as.numeric(final$speech_year)
final <- final %>% arrange(speech_year)

# Separate Senate and House data
sen <- filter(final, chamber == "senate")
con <- filter(final, chamber == "house")

p <- ggplot() +
  # Senate plot with fill for standard error
  geom_line(data = sen, aes(x = speech_year, y = mean_score), color = "red", linewidth = 1.5) +
  geom_ribbon(data = sen, aes(x = speech_year, ymin = mean_score - std_error, ymax = mean_score + std_error), fill = "red", alpha = 0.3) +
  
  # House plot with fill for standard error
  geom_line(data = con, aes(x = speech_year, y = mean_score), color = "green", linewidth = 1.5) +
  geom_ribbon(data = con, aes(x = speech_year, ymin = mean_score - std_error, ymax = mean_score + std_error), fill = "green", alpha = 0.3) +
  
  # Historical periods
  annotate("rect", xmin = 1861, xmax = 1865, ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.1) +
  annotate("text", x = 1860, y = 0.98, label = "Civil War", size = 6, color = "black") +
  
  annotate("rect", xmin = 1914, xmax = 1918, ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.1) +
  annotate("text", x = 1913, y = 0.99, label = "WW1", size = 6, color = "black") +
  
  annotate("rect", xmin = 1939, xmax = 1945, ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.1) +
  annotate("text", x = 1939, y = 1.02, label = "WW2", size = 6, color = "black") +
  
  annotate("rect", xmin = 1979, xmax = 1986, ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.1) +
  annotate("rect", xmin = 1986, xmax = 2015, ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.2) +
  
  # Labels and formatting
  labs(title = "Emotionality in U.S. Congress by Chamber", 
       x = "Year", 
       y = "Mean Emotionality Score") +
  theme_minimal() +
  theme(
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 90, hjust = 1),
    legend.position = "top"
  ) +
  scale_x_continuous(breaks = seq(1858, 2015, 5)) +
  annotate("text", x = 1987, y = 1.1, label = "C-SPAN2", size = 6) +
  annotate("text", x = 1977, y = 1.07, label = "C-SPAN1", size = 6)

print(p)


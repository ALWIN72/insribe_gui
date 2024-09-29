import os
from groq import Groq
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox


# Initialize the Groq API client
client = Groq()


class QuizGameGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("INSCRIBE")
        self.window.geometry("800x600")

        self.topic_label = tk.Label(self.window, text="", wraplength=700, font=("Arial", 14))
        self.topic_label.pack(pady=20)

        self.question_label = tk.Label(self.window, text="", wraplength=700, font=("Arial", 14))
        self.question_label.pack(pady=20)

        self.topic_entry = tk.Entry(self.window, width=50, font=("Arial", 14))
        self.topic_entry.pack(pady=10)

        self.generate_button = tk.Button(self.window, text="Generate Quiz", command=self.generate_quiz)
        self.generate_button.pack(pady=10)

        self.answer_entry = tk.Text(self.window, height=10, width=50, font=("Arial", 14))
        self.answer_entry.pack(pady=20)

        self.check_button = tk.Button(self.window, text="Check Answer", command=self.check_answer)
        self.check_button.pack(pady=10)

        self.hint_label = tk.Label(self.window, text="", wraplength=700, font=("Arial", 14))
        self.hint_label.pack(pady=10)

        self.score_label = tk.Label(self.window, text="Score: 0", font=("Arial", 14))
        self.score_label.pack(pady=10)
        
        self.knowledge_label = tk.Label(self.window, text="Knowledge Percentage: 0.00%", font=("Arial", 14))
        self.knowledge_label.pack(pady=10)
        
        self.vocabulary_label = tk.Label(self.window, text="", wraplength=700, font=("Arial", 14))
        self.vocabulary_label.pack(pady=10)

        self.difficulty_level = 1
        self.score = 0
        
        self.quit_button = tk.Button(self.window, text="Quit Game", command=self.window.destroy)
        self.quit_button.pack(pady=10)

    def generate_quiz(self):
        user_topic = self.topic_entry.get()
        self.topic, self.question = self.generate_topic_and_question(user_topic, self.difficulty_level)
        self.correct_answer = self.generate_correct_answer(self.topic)

        self.topic_label['text'] = f"Topic: {self.topic}"
        self.question_label['text'] = f"Question: {self.question}"

    def generate_topic_and_question(self, prompt, difficulty_level):
        try:
            difficulty_descriptions = {
                1: "easy",
                2: "medium",
                3: "hard"
            }
            
            difficulty_description = difficulty_descriptions.get(difficulty_level, "easy")
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate a {difficulty_description} topic and a question for short essay about: {prompt}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            response_text = response.choices[0].message.content.strip()
            topic, question = None, None
            
            if "Topic:" in response_text and "Question:" in response_text:
                try:
                    topic = response_text.split("Topic:")[1].split("\n")[0].strip()
                    question = response_text.split("Question:")[1].split("\n")[0].strip()
                    
                    if not topic or not question:
                        raise ValueError("Topic or question is empty.")
                        
                except (IndexError, ValueError) as e:
                    print(f"Error parsing response: {e}")
                    topic, question = None, None
            else:
                print("Couldn't generate topic and question. Please try again.")
            
            return topic, question
        
        except Exception as e:
            messagebox.showerror("Error", f"Error generating topic and question: {e}")
            return None, None


    def generate_correct_answer(self, topic):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Provide a short essay about: {topic}"
                    }
                ],
                temperature=0.7,
                max_tokens=150,
                top_p=1,
                stream=False,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if not response_text:
                messagebox.showerror("Error", "No answer generated. Please try again.")
                return None
            
            return response_text
        
        except Exception as e:
            messagebox.showerror("Error", f"Error generating correct answer: {e}")
            return None


    def vocabulary_analysis(self, text):
        try:
            # Use Groq API to analyze vocabulary
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze vocabulary skills in: {text}, provide feedback"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            response_text = response.choices[0].message.content.strip()
            feedback = ""
            
            if "Vocabulary size:" in response_text:
                vocabulary_size_feedback = response_text.split("Vocabulary size:")[1].split("\n")[0].strip()
                feedback += vocabulary_size_feedback + "\n"
            
            if "Sentence structure:" in response_text:
                sentence_structure_feedback = response_text.split("Sentence structure:")[1].split("\n")[0].strip()
                feedback += sentence_structure_feedback + "\n"
            
            if "Complex word usage:" in response_text:
                complex_word_usage_feedback = response_text.split("Complex word usage:")[1].split("\n")[0].strip()
                feedback += complex_word_usage_feedback + "\n"
            
            return feedback
        
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing vocabulary: {e}")
            return ""


    def check_answer(self):
        user_answer = self.answer_entry.get("1.0", "end-1c")
        # Check grammar and spelling
        error_message = ''
        grammar_response, grammar_errors = self.check_grammar(user_answer)
        spelling_response, spelling_errors = self.check_spelling(user_answer)

        if grammar_response is None or spelling_response is None:
            messagebox.showerror("Error", "Error checking grammar or spelling.")
            return
        
        if grammar_errors or spelling_errors:
            if grammar_errors:
                error_message += "Grammar errors:\n"
                for error in grammar_errors:
                    error_message += f"- {error}\n"
            if spelling_errors:
                error_message += "\nSpelling errors:\n"
                for error in spelling_errors:
                    error_message += f"- {error}\n"
        
            response = messagebox.askyesno("Errors Found", error_message + "\nWould you like to correct these errors?")
        
            if response:
                corrected_answer = self.correct_errors(user_answer, grammar_errors, spelling_errors)
                self.answer_entry.delete('1.0', tk.END)
                self.answer_entry.insert('1.0', corrected_answer)


        # Check user's knowledge percentage
        knowledge_percentage = self.knowledge_check(user_answer, self.correct_answer)
        vocabulary_feedback = self.vocabulary_analysis(user_answer)
        
        # Display feedback
        if vocabulary_feedback:
            self.vocabulary_label['text'] = "Vocabulary Feedback:\n" + vocabulary_feedback
        else:
            self.vocabulary_label['text'] = "No vocabulary feedback available."
        
        # Display knowledge percentage
        self.knowledge_label['text'] = f"Knowledge Percentage: {knowledge_percentage:.2f}%"
        
        # Set similarity thresholds based on the level
        if self.difficulty_level == 1:
            similarity_threshold = 0.2  
        elif self.difficulty_level == 2:
            similarity_threshold = 0.5  
        elif self.difficulty_level == 3:
            similarity_threshold = 0.7  
    
        if self.cosine_similarity_check(self.correct_answer, user_answer) > similarity_threshold:
            self.score += 1
            self.score_label['text'] = f"Score: {self.score}"
            self.difficulty_level += 1  
            if self.difficulty_level > 3:
                messagebox.showinfo("Quiz Complete", "Congratulations! You've completed the quiz.")
                self.window.destroy()
            else:
                self.topic, self.question = self.generate_topic_and_question(self.topic, self.difficulty_level)
                self.correct_answer = self.generate_correct_answer(self.topic)
                self.topic_label['text'] = f"Topic: {self.topic}"
                self.question_label['text'] = f"Question: {self.question}"
                messagebox.showinfo("Correct", "Correct! You earned points.")
        else:
            self.provide_hints(self.correct_answer, user_answer)
            messagebox.showinfo("Incorrect", "Incorrect. Try again!")


    def correct_errors(self, text, grammar_errors, spelling_errors):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Correct grammar and spelling errors in: {text}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            corrected_text = response.choices[0].message.content.strip()
            return corrected_text
        
        except Exception as e:
            messagebox.showerror("Error", f"Error correcting errors: {e}")
            return text


    def knowledge_check(self, user_answer, correct_answer):
        similarity_ratio = self.cosine_similarity_check(correct_answer, user_answer)
        knowledge_percentage = similarity_ratio * 100  
        return knowledge_percentage


    def cosine_similarity_check(self, correct_answer, user_answer):
        vectorizer = CountVectorizer().fit_transform([correct_answer, user_answer])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0][1]  


    def provide_hints(self, correct_answer, user_answer):
        similarity_ratio = self.cosine_similarity_check(correct_answer, user_answer)
        
        if similarity_ratio > 0.7:  
            self.hint_label['text'] = "You're very close! Consider refining your answer."
        elif similarity_ratio > 0.4:
            self.hint_label['text'] = "Good effort! Try to include more relevant details."
        else:
            self.hint_label['text'] = "It seems like your answer could be more aligned with the question. Revisit the topic for better understanding."


    def check_grammar(self, text):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Check for grammar in: {text}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            if response:
                response_text = response.choices[0].message.content.strip()
                errors = []
                if "Error" in response_text or "error" in response_text.lower():
                    error_messages = response_text.split("\n")
                    for error in error_messages:
                        if "Error" in error or "error" in error.lower():
                            errors.append(error.strip())
                return response, errors
            else:
                return None, []
        
        except Exception as e:
            messagebox.showerror("Error", f"Error checking grammar: {e}")
            return None, [str(e)]


    def check_spelling(self, text):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"Check for spelling in: {text}, provide suggestions"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            if response:
                response_text = response.choices[0].message.content.strip()
                errors = []
                if "Error" in response_text or "error" in response_text.lower():
                    error_messages = response_text.split("\n")
                    for error in error_messages:
                        if "Error" in error or "error" in error.lower():
                            errors.append(error.strip())
                return response, errors
            else:
                return None, []
        
        except Exception as e:
            messagebox.showerror("Error", f"Error checking spelling: {e}")
            return None, [str(e)]


    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = QuizGameGUI()
    gui.run()
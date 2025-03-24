from groq import Groq

class Groq_Agent:
    def __init__(self, groq_api_key, model="gemma2-9b-it"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_api_key = groq_api_key
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def make_query(self, query, system_prompt = "You are Sam, a helpful friend who wants the best for the current user."):
        """
        Makes a query to the Groq API and returns the (num_input_tokens, response text, num_output_tokens)
        Also tracks token usage.
        """
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}],
                model=self.model
            )
            
            # Update token counts
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            
            return (response.usage.prompt_tokens, response.choices[0].message.content, response.usage.completion_tokens)

        except Exception as e:
            print(f"Error making Groq query: {str(e)}")
            # If possible, find a better alternative than just setting input prompt tokens to 0
            return (0 , None, 0)

    def get_token_usage(self):
        """
        Returns the current token usage statistics
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens
        }
from learning_module import Learner
from retrieval_module import Memory
from groq_interface import Groq_Agent
from collections import deque
import os

# Processing module
class Conversationalist:
    def __init__(self, context_limit = 5000, model="") -> None:
        # Init for this module itself
        self.current_convo_context = deque();
        self.current_convo_context_size = 0 # How many characters we have in our current convo context
        self.convo_context_size_limit = context_limit # how many characters we can have in our context size
        self.context_string = ""


        # Init for memory
        self.memory = Memory()
        self.user_name = self.memory.get_basic_info("name") 
        # TODO: Add error checking in case we don't know name
        self.date = self.memory.get_date() # returned in string format

        # Init for groq interface
        self.llm = Groq_Agent(groq_agent = Groq_Agent(os.getenv("GROQ_API_KEY"), model="qwen-2.5-32b"))

        # Init for learner
        self.learner = Learner(self.memory,self.llm)

    def _get_convo_context(self, actor, query):
        # prompt to extract context given previous context
        prompt = f"""
        Context up till now: {self.context_string}\n, 
        Using this, extract all useful information(be thorough, do not skip any useful information) from the newest query: {query}\n,
        and return it in this format, 
        <context>
        .....
        <\context>
        FOR YOUR SAKE, do not return any extra text outside the <context> clause!!!
        """
        (_, str_resp, size) = self.llm.make_query(prompt)
        # Extract context between tags from LLM response
        if str_resp:
            try:
                # Find content between context tags
                start_idx = str_resp.find("<context>") + len("<context>")
                end_idx = str_resp.find("<\\context>")
                if start_idx >= 0 and end_idx >= 0:
                    context_summary = str_resp[start_idx:end_idx].strip()
                    # Return tuple in format (actor, summary, size) as specified in convo_context_example.txt
                    return (actor, context_summary, len(context_summary))
            except Exception as e:
                print(f"Error extracting context: {str(e)}")
        
        # Return empty context if extraction failed
        return (actor, "", 0)
    
    def _add_convo_context(self, new_context):
        # Get only as much context as you have space for
        if self.current_convo_context_size + new_context[2] <= self.convo_context_size_limit:
            self.current_convo_context.append(new_context)
            self.current_convo_context_num += 1
            self.current_convo_context_size += new_context[2]
            # return 1
        else:
            # out of space, shed earlier context
            while self.current_convo_context_size > self.convo_context_size_limit:
                self.current_convo_context_size -= self.current_convo_context.popleft()[2]
            # return 0
        
    def _return_context(self):
        out_string = ""
        for (actor, info, _) in self.current_convo_context:
            out_string += f"{actor}:{info}\n"
        return out_string

    def process_query(self, user_query):
        # Pass query and user context to learning module so it can learn from the user's responses
        # Current convo context is an ordered list of extracted info from earlier parts of the convo
        # Send computed new context to llm
        new_context = self._get_convo_context(self.user_name, user_query)
        self._add_convo_context(new_context)
        learning_stuff = f"""
        Summary of conversation so far:
        {self.context_string}

        User's last interaction: {user_query}
        """
        extra_context_string = self.memory.retrieve(learning_stuff) # improperly named, but whatever
        self.learner.learn_from_query(learning_stuff, extra_context_string)
        finalized_prompt = f"""
        Summary of conversation so far:
        {self.context_string}

        Retrieved context related to query:
        {extra_context_string}

        User's last interaction: {user_query}
        Please return a friendly and helpful response
        """

        self.context_string = self._return_context
        _, response, _ = self.llm.make_query(finalized_prompt)
        self.memory.write_to_disk() # write to disk at the end of each query
        return response
from learning_module import Learner
from retrieval_module import Memory
from groq_interface import Groq_Agent
from collections import deque
import re
import os

# Processing module
class Conversationalist:
    def __init__(self, context_limit = 20000, model="llama3-70b-8192", groq_api_key = os.getenv("GROQ_API_KEY")) -> None:
        # Init for this module itself
        self.current_convo_msgs_num = 0
        self.current_convo_context = deque();
        self.current_convo_context_size = 0 # How many characters we have in our current convo context
        self.convo_context_size_limit = context_limit # how many characters we can have in our context size
        self.context_string = ""

        # Init for groq interface
        self.llm = Groq_Agent(os.getenv("GROQ_API_KEY"), model=model)

        # Init for memory
        self.memory = Memory(self.llm)
        self.user_name = self.memory.get_basic_info("name") 
        self.preferred_lang = self.memory.get_basic_info("preferred_lang")
        self.other_langs = self.memory.get_basic_info("alt_langs")
        self.llm.change_lang(self.preferred_lang)
        
        # context from last convo
        self.general_convo_start_info = self.memory.convo_start_info["general_info"]
        self.recent_context = self.memory.convo_start_info["prev_context"]

        # TODO: Add error checking in case we don't know name
        self.date = self.memory.get_date() # returned in string format

        # Init for learner
        self.learner = Learner(self.memory,self.llm)

    def _get_convo_context(self, actor, query):
        # prompt to extract context given previous context
        prompt = f"""
        Context up till now: {self.context_string}\n, 
        Using this, extract all useful information(be concise, but do not skip any imp. information) from the newest query:\n {query}\n,
        and return it in this format, 
        <CT>
        .....
        <CT>
        FOR YOUR SAKE, do not return any extra text outside the <context> clause!!!
        Be concise, feel free to return an empty string if you feel there's no useful information
        """
        (_, resp, size) = self.llm.make_query(prompt, temp=0.2)
        # Extract context between tags from LLM response
        if resp:
            try:
                # Find content between context tags
                context_str = re.search(r"<CT>(.*?)<CT>",resp,re.DOTALL)
                if context_str is None:
                    context_str = re.search(r"<CT>(.*?)<\\CT>",resp,re.DOTALL)
                if context_str is None:
                    context_str = re.search(r"<CT>(.*?)<\\\\CT>",resp,re.DOTALL)
                if context_str is not None:
                    context_summary = context_str.group(1).strip()
                    # Return tuple in format (actor, summary, size) as specified in convo_context_example.txt
                    return (actor, context_summary, len(context_summary))
            except Exception as e:
                print(f"Error extracting context: {str(e)}")
        
        # Return empty context if extraction failed
        return (actor, "", 0)
    
    def _add_convo_context(self, new_context):
        # Get only as much context as you have space for
        if new_context[1].strip() == "":
            # Don't bother adding negligible context
            return
        if self.current_convo_context_size + new_context[2] <= self.convo_context_size_limit:
            self.current_convo_context.append(new_context)
            self.current_convo_msgs_num += 1 # Keep track of how many significant messages have come in the convo so far
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

    def start_convo(self):
        # Just starting the convo, use convo start general info and prev context
        prompt = f"""
            Here's some general info about {self.user_name}:
            {self.general_convo_start_info}
            and here's what you guys spoke about last time:
            {self.recent_context}
            Most important thing: BE COOL, adapt to the guy
            Please respond as if you were the user's personal friend, 
            from the 2nd person perspective as if talking directly to the user.
            Feel free to ignore any useless information. 
            For simple interactions, like a simple hi, just say hi to the user.
            Don't be verbose, be cool, talk as if you were actually his friend. 
            Try to mimic normal human interaction as far as you can, be platonic and absolutely no romantic or sexual references.
            Also, you're not a physical human being, so you know, don't talk about going out to drinks or whatever
        """
        (_,hello_msg, _) = self.llm.make_query(prompt, temp=0.9)
        return hello_msg

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
        self.last_learned_thing = self.learner.learn_from_query(learning_stuff, extra_context_string)
        finalized_prompt = f"""
        You're having a casual conversation with someone:
        What they said last: "{user_query}"
        Now, for context:
        Summary of conversation so far:
        {self.context_string}

        Retrieved context related to query:
        {extra_context_string}

        Instructions:
        1. CONTINUE the conversation as their personal friend, try to mimic normal human interaction as far as you can.
        2. Talk from the 2nd person POV, talking directly to the user.
        3. Feel free to ignore any useless information. And feel more than free to ask when you don't know. 
        4. You don't have to respond to everything he says!!
        5. Don't be verbose, be cool.
        6. Also, you're not a physical human being, so you know, don't talk about going out to drinks or whatever
        """

        self.context_string = self._return_context()
        _, response, _ = self.llm.make_query(finalized_prompt, temp=0.7)
        self.memory.write_to_disk() # write to disk at the end of each query
        return response
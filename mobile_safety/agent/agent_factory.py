from mobile_safety.agent.qwen_agent import QwenAgent
from mobile_safety.agent.gpt_agent import GPTAgent
from mobile_safety.agent.gemini_agent import GeminiAgent
from mobile_safety.agent.guard import FilterAgent, PredictorAgent
from mobile_safety.prompt._prompt import PromptBuilder
from mobile_safety.agent.models import ModelWrapper

import cv2
from PIL import Image
import os
import datetime
import copy 

_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]

class WrappedAgent:
    '''
    wrap Agent
    '''
    def __init__(self, args):
        self.args = args
        self.agent_model = args.agent_model
        self.prompt_mode = args.prompt_mode
        self.mode = args.mode
        self.guard_model = args.guard_model

        self.with_helpful = args.with_helpful

        self.filter = None
        self.planner = None
        self.predictor = None

        self.seed = args.seed
        self.port = args.port
        self.task = None

        self.max_retry = 1

        self._task_filtered = False  # Track if task has been filtered

        self._init_agent()

    # init agent according to args
    def _init_agent(self):
        # init planner agent
        if self.planner == None:
            if 'gpt' in self.agent_model:
                self.planner = GPTAgent(model_name=self.agent_model, seed=self.seed, port=self.port)
            elif 'gemini' in self.agent_model:
                self.planner = GeminiAgent(model_name=self.agent_model, seed=self.seed, port=self.port)
            elif 'Qwen' in self.agent_model or 'qwen' in self.agent_model:
                self.planner = QwenAgent(model_name=self.agent_model, seed=self.seed, port=self.port)

        # init filter agent
        if self.mode in ['guard', 'filter']:
            self.filter = FilterAgent(ModelWrapper(self.guard_model))

        # init predictor agent
        if self.mode in ['guard', 'predictor']:
            self.predictor = PredictorAgent(ModelWrapper(self.guard_model))

        print('Finished Initializing Agent.')

    def get_context(self):
        if self.planner:
            return self.planner.context

    def step(self, timpstep, env):
        '''
        return 
            response_dict, prompt, time_consumption
        '''

        # Extract task from environment
        if self.task is None:
            self.task = getattr(env, 'instruction', '')


        if not hasattr(self, 'prompt_builder'):
            self.prompt_builder = PromptBuilder(env)

        # need to processed by filter
        if self.filter and not self._task_filtered:
            self._task_filtered = True
            filter_result, _, _ = self.filter.check(self.task)

            #print(f"Agent Mode {self.mode}, FiterAgent Response: {filter_result}")

            if filter_result.get("safety-judge", 'safe') == 'unsafe':
                print(f"Task rejected by SeerGuard-Instruction-Level: {filter_result.get('safety-reason', 'Unknown reason')}")
                return {
                    "action": "refuse()",
                    "raw_response": f"Task rejected: {filter_result.get('safety-reason', 'This task is not allowed.')}"
                }, None, datetime.timedelta(0)


        if not self.predictor:
            prompt = self.prompt_builder.build(
				parsed_obs=env.parsed_obs,
				action_history=env.evaluator.actions[1:],
				action_error=env.action_error,
			)

            response_dict, time_elapsed = self.planner.get_response(timestep=timpstep, system_prompt=prompt.system_prompt, user_prompt=prompt.user_prompt)
            return response_dict, prompt, time_elapsed
        

        else:
            prompt = self.prompt_builder.build(
				parsed_obs=env.parsed_obs,
				action_history=env.evaluator.actions[1:],
				action_error=env.action_error,
			)

            # Save current step's base context
            step_context = copy.copy(self.planner.context)

            ## Use context for safety-related reflection
            reflection = []
            total_time_elapsed = datetime.timedelta(0)
            screenshot_path = self.save_image(timestep=timpstep)

            for try_num in range(1, self.max_retry + 1):
                print(f"[Agent] Received task: {self.task}")
                self.planner.context = step_context+'\n'+'\n'.join(reflection)
                #print(f"{try_num} try: Agent Mode {self.mode}, PlannerAgent context: {self.planner.context}")
                response_dict, tmp_time_elapsed = self.planner.get_response(timestep=timpstep, system_prompt=prompt.system_prompt, user_prompt=prompt.user_prompt)
                total_time_elapsed += tmp_time_elapsed

                #print(f"{try_num} try: Agent Mode {self.mode}, PlannerAgent output: {response_dict}")
                print(f"[Agent] Response: {response_dict}")

                action_str = str(response_dict['action'])

                # Validate with PredictorAgent
                prediction, _, _ = self.predictor.predict(
                    task=self.task,
                    action=action_str,
                    images=[screenshot_path],
                    parsed_obs=env.parsed_obs
                )
                #print(f"{try_num} try: Agent Mode {self.mode}, PredictorAgent output: {prediction}")

                if prediction.get("safety-judge", 'SAFE') == 'SAFE':
                    if self.with_helpful:
                        if prediction.get("effectiveness-judge", 'YES') == 'YES':
                            # Action is safe and helpful
                            return response_dict, prompt, total_time_elapsed
                        else:
                            reflection.append(f"Try {try_num}: Action unhelpful - {action_str}")
                    else:
                        return response_dict, prompt, total_time_elapsed
                else:
                    reflection.append(f"Try {try_num}: Action unsafe - {prediction.get('safety-reason', '')}")

            # All retries failed - refuse to act
            print(f"All {self.max_retry} retries failed. Refusing to act.")
            return {"action": "refuse()", "raw_response": "Unable to find safe/helpful action"}, None, total_time_elapsed
        
    def save_image(self, timestep=None):
        img_obs = timestep.curr_obs["pixel"]
        img_cv = cv2.resize(img_obs, dsize=(1024, 2048), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_cv)
        img_pil_path = f"{_WORK_PATH}/logs/tmp_{self.port}.png"
        img_pil.save(img_pil_path)

        return img_pil_path








        

    
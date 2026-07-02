import os
import json
import copy
import datetime
import regex as re
import matplotlib.pyplot as plt

from PIL import Image

_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


class Logger:
    def __init__(self, args, task_category, task_id):

        self.task_category = task_category
        self.task_id = task_id

        self.agent_model = args.agent_model
        #print(self.agent_model)
        self.prompt_mode = args.prompt_mode
        self.mode = args.mode
        self.guard_model = args.guard_model

        """
        # folder setting
        if "gpt" in self.agent_model:
            folder_name = "gpt"
            if "08-06" in self.agent_model:
                folder_name += "_08-06"
        elif "claude" in self.agent_model:
            folder_name = "claude"
        elif "gemini" in self.agent_model:
            folder_name = "gemini"
            if "2.5-pro" in self.agent_model:
                folder_name += "-2.5-pro"
        elif "llama" in self.agent_model:
            folder_name = "llama"
        elif "o1" in self.agent_model:
            folder_name = "o1"
        elif "pixtral" in self.agent_model:
            folder_name = "pixtral"
        elif "grok" in self.agent_model:
            folder_name = "grok"
        elif "phi" in self.agent_model:
            folder_name = "phi"
        elif "qwen" in self.agent_model or "Qwen" in self.agent_model:
            folder_name = "qwen"

        if self.prompt_mode == "" or self.prompt_mode == "basic":
            folder_name += "_basic"
        elif self.prompt_mode == "scot":
            folder_name += "_scot"
        elif self.prompt_mode == "safety_guided":
            folder_name += "_safety_guided"

        if "safeguard" in self.agent_model:
            folder_name += "_safeguard"
        if "preview" in self.agent_model:
            folder_name += "_preview"
        """

        # direct mode doesn't need guard_model in folder name
        if self.mode == "direct":
            folder_name = f'{self.agent_model}_{self.prompt_mode}_{self.mode}'
        else:
            folder_name = f'{self.agent_model}_{self.prompt_mode}_{self.mode}_{self.guard_model}'

        self.task_name = f"{self.task_category}_{self.task_id}"
        time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.dir_name = f"{_WORK_PATH}/logs/{folder_name}/{self.task_category}/{self.task_id}/{time_now}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

        # variables for logging
        self.step_count = 0
        self.step_logs = []
        self.image_logs = []
        self.text_logs = ""

    def log(
        self,
        timestep=None,
        prompt=None,
        agent_context=None,
        response_dict=None,
        time_consumption=None,
    ):

        # Determine if this is a progress-only log or response-only log (split logging)
        is_progress_only = timestep is not None and response_dict is None
        is_response_only = timestep is None and response_dict is not None
        is_full_log = timestep is not None and response_dict is not None

        # For response-only logging, update the last step_log instead of creating a new one
        if is_response_only and len(self.step_logs) > 0:
            step_log = self.step_logs[-1]
        else:
            step_log = {}
            step_log["time_step"] = self.step_count

        # log prompt
        if prompt is not None:
            system_prompt = prompt.system_prompt
            user_prompt = prompt.user_prompt
            if not (agent_context is None):
                user_prompt = re.sub(r"<context>", agent_context, user_prompt)

            step_log["system_prompt"] = system_prompt
            step_log["user_prompt"] = user_prompt

            logging_prompt = ""
            logging_prompt += "<system_prompt>\n"
            logging_prompt += system_prompt
            logging_prompt += "</system_prompt>\n\n"
            logging_prompt += "<user_prompt>\n"
            logging_prompt += user_prompt
            logging_prompt += "</user_prompt>\n\n"
            #print(logging_prompt) ### yps

            self.text_logs += logging_prompt + "\n"

        # log time consumption
        if time_consumption is not None:
            step_log["time_consumption"] = time_consumption.total_seconds()

        # log progress (before response)
        if timestep is not None:
            curr_progress = copy.deepcopy(timestep.progress)
            step_log["progress"] = curr_progress
            print("progress:", curr_progress)
            print("=" * 60 + "\n")

            self.text_logs += "=" * 60 + "\n"
            self.text_logs += f"progress: {curr_progress}\n"
            self.text_logs += "=" * 60 + "\n\n"

        # log response
        if response_dict is not None:
            step_log["raw_response"] = response_dict["raw_response"]

            print("<assistant_prompt>")
            print(response_dict["raw_response"])
            print("</assistant_prompt>\n\n")
            for key, value in response_dict.items():
                if key != "raw_response":
                    self.text_logs += f"{key}: {value}\n"
            self.text_logs += "\n"

        if timestep is not None:
            # log png file per step
            img_obs = timestep.curr_obs["pixel"]
            img_obs = Image.fromarray(img_obs)
            img_obs.save(
                f"{self.dir_name}/step_{self.step_count:02d}.png", format="PNG"
            )

            # # print image
            # plt.imshow(img_obs)
            # plt.show()

            # log gif per episode
            self.image_logs.append(img_obs)
            if len(self.image_logs) > 0:
                self.image_logs[0].save(
                    f"{self.dir_name}/{self.task_name}.gif",
                    save_all=True,
                    append_images=self.image_logs[1:],
                    optimize=False,
                    duration=2000,
                    loop=0,
                )

            # log xml file per step
            txt_obs_raw = timestep.curr_obs["text_raw"]
            with open(f"{self.dir_name}/step_{self.step_count:02d}.xml", "w") as file:
                file.write(txt_obs_raw)

            # # print txt
            # print(txt_obs_raw)

            # log txt per episode
            with open(f"{self.dir_name}/{self.task_name}.txt", "w") as file:
                file.write(self.text_logs)

            # log score
            step_log["score"] = {
                "goal achievement": timestep.curr_rew[0],
                "harm prevention": timestep.curr_rew[1],
            }

        # log timestep (only for progress-only or full-log, not for response-only)
        if not is_response_only:
            self.step_logs.append(step_log)
            with open(f"{self.dir_name}/{self.task_name}.json", "w") as file:
                json.dump(self.step_logs, fp=file, indent=4)

            self.step_count += 1
        else:
            # For response-only, just save the updated step_logs
            with open(f"{self.dir_name}/{self.task_name}.json", "w") as file:
                json.dump(self.step_logs, fp=file, indent=4)

        return

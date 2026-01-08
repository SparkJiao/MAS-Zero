from common import ANSWER_PATTERN, async_check_equality
from sampler import AsyncChatCompletionSampler

from utils import extract_xml
import common
import json
from common import HTML_JINJA, SingleEvalResult
import re


class DataScorer:

    def __init__(self, dataset, technique, mode_verifier):
        self.dataset = dataset
        self.technique = technique
        self.equality_checker = AsyncChatCompletionSampler(model="gpt-4-turbo-preview")
        self.mode_verifier = mode_verifier
        self.LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def _is_swe_dataset(self):
        return any(tag in self.dataset for tag in ('swe_bench', 'workflow_search/swe', 'swe_test'))

    async def run_score(self, answer, extracted_answer, use_oracle_verifier, judge_path, instance_id, n, code_snippet):

        if self._is_swe_dataset():
            print("SWE verification placeholder: returning 0.0 (requires offline evaluation).")
            return 0.0
        elif 'aime24' in self.dataset or 'hle_math' in self.dataset:
            res = await async_check_equality(self.equality_checker, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(res)
        elif 'browsecomp-plus' in self.dataset:
            res = await async_check_equality(self.equality_checker, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(res)
        elif 'gpqa_diamond' in self.dataset:
            res = extracted_answer
            is_early_stop = False
            try:
                if isinstance(res, str) and res in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res]
                elif 'A)' in res:
                    predicted_idx = 0
                elif 'B)' in res:
                    predicted_idx = 1
                elif 'C)' in res:
                    predicted_idx = 2
                elif 'D)' in res:
                    predicted_idx = 3
                elif isinstance(res, list):
                    try_res = res[1]
                    predicted_idx = self.LETTER_TO_INDEX[try_res.content]
                elif res.content in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res.content]
                elif 'A)' in res.content:
                    predicted_idx = 0
                elif 'B)' in res.content:
                    predicted_idx = 1
                elif 'C)' in res.content:
                    predicted_idx = 2
                elif 'D)' in res.content:
                    predicted_idx = 3
                else:
                    print(f"error in q {instance_id}")
                    score = 0
                    is_early_stop = True
            except Exception as e:
                score = 0
                is_early_stop = True

            if not is_early_stop:  # if cannot find predicted_idx, then done
                if predicted_idx == answer:
                    score = 1
                else:
                    score = 0

            print(f'extracted_answer: {extracted_answer}; answer: {answer}; score: {score}')

            return score
        elif 'folio' in self.dataset:
            res = extracted_answer
            if 'True' in res:
                pred = 'True'
            elif 'False' in res:
                pred = 'False'
            elif 'Uncertain' in res:
                pred = 'Uncertain'
            else:
                pred = ''

            score = pred == answer
            return float(score)
        elif 'knights-and-knaves' in self.dataset:
            _judge_prompt = ("I will show you a model's response as well as the ground-truth answer towards a knights-and-knaves puzzle. "
                             "Please determine if the model's response is consistent with the ground truth."
                             "\n\nMode's Response:\n"
                             "{response}\n\n"
                             "Ground Truth:\n"
                             "{answer}\n\n"
                             "Your response should only contains `Yes` or `No`.").format(response=extracted_answer, answer=answer)

            res = await self.equality_checker([{"role": "user", "content": _judge_prompt}], response_format="normal")
            res = res[0]
            if "yes" in res.lower():
                return 1.0
            return 0.0
        elif 'hanoi' in self.dataset:
            from hanoi import judge_prompt

            _judge_prompt = ("Here is a move sequence of Hanoi Game:\n\n{response}\n\n"
                             "Please evaluate the it according to the following criteria:\n\n") + judge_prompt

            _judge_prompt = _judge_prompt + "\n\nYou can first think step by step, and put your final decision in <decision> True or False </decision>."

            res = await self.equality_checker([{"role": "user", "content": _judge_prompt}], response_format="normal")
            res = res[0]
            m = re.search(r"<decision>\s*(true|false)\s*</decision>", res, re.IGNORECASE)
            # return None if not m else (m.group(1).lower() == "true")
            if not m:
                return 0.0
            pred = m.group(1).lower()
            return float(pred == "true")
        else:
            raise NotImplementedError

    async def score(self, example_id, n, prompt_message, question, response_text, answer, sub_tasks_text, use_oracle_verifier, judge_path, response_path,
                    response_dict, instance_id, code_snippet):

        if self._is_swe_dataset():
            extracted_answer = response_text.split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()
        else:
            try:
                match = re.search(ANSWER_PATTERN, response_text)
                extracted_answer = match.group(1) if match else None
                extracted_answer = extracted_answer.strip()
            except NameError as e:
                import traceback
                traceback.print_exc()
                print(ANSWER_PATTERN)
                print(response_text)
                raise e

        print('extracted_answer: ', extracted_answer)

        with open(judge_path, 'a+') as judge_file:
            judge_file.write(f'Question: {question}\nproposed answer: {response_text}\nExtracted answer: {extracted_answer}\nCorrect answer: {answer}\n')

        with open(response_path, 'w') as json_file:
            response_dict.append({
                'example_id': example_id,
                'problem': question,
                'correct_answer': answer,
                'n': n,
                'response': response_text,
                'sub_tasks_text': sub_tasks_text})

            json.dump(response_dict, json_file, indent=4)

        if use_oracle_verifier:
            score_oracle_verifier = await self.run_score(answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path, instance_id=instance_id,
                                                         n=n,
                                                         code_snippet=code_snippet)
            score = score_oracle_verifier
            score_model_verifier = None
        else:
            if sub_tasks_text is None:
                score_model_verifier = await self.run_score(self.mode_verifier, question, response_text, use_oracle_verifier=False, judge_path=judge_path,
                                                            instance_id=instance_id, n=n, code_snippet=code_snippet)
            else:
                score_model_verifier = await self.run_score(self.mode_verifier, question, sub_tasks_text, use_oracle_verifier=False, judge_path=judge_path,
                                                            instance_id=instance_id, n=n, code_snippet=code_snippet)
            score = score_model_verifier

        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=prompt_message,
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=answer,
            extracted_answer=extracted_answer,
        )
        convo = prompt_message + [dict(content=response_text, role="assistant")]
        results = SingleEvalResult(html=html, score=score, convo=convo)
        return score_oracle_verifier, score_model_verifier, results

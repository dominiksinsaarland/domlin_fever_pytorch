# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import six
import random
import re
import tokenization

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):


    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = fever_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        #print ("clue_convert_examples max length", max_length)
        #input("")
        #return input_ids, input_mask, segment_ids
	#def convert_single_example(ex_index, text_a, text_b, max_seq_length, tokenizer):


        input_ids, attention_mask, token_type_ids = tokenization.convert_single_example(ex_index, example.text_a, example.text_b, max_length, tokenizer)
        """
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=True  # We're truncating the first sequence in priority
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        if len(input_ids) != max_length:
            print ("somethings wrong")
            print (example)
            print (example.text_a, example.text_b)
        """
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features



class FirstSentenceRetrievalProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, filename=None):
        """See base class."""

        if filename is not None:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "train", 4)

        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "sentence_retrieval_1_training_set.tsv")))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "sentence_retrieval_1_training_set.tsv")), "train", 4)

    def get_dev_examples(self, data_dir, filename=None):
        """See base class."""
        if filename is not None:        
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "dev", 0)
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "sentence_retrieval_1_dev_set.tsv")))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "sentence_retrieval_1_dev_set.tsv")), "dev", 0)

    def get_labels(self):
        """See base class."""
        return [None]

    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def process_evid(self, sentence):
        sentence = re.sub(" -LSB-.*-RSB-", " ", sentence)
        sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
        sentence = re.sub("-LRB-", "(", sentence)
        sentence = re.sub("-RRB-", ")", sentence)
        sentence = re.sub("-COLON-", ":", sentence)
        sentence = re.sub("_", " ", sentence)
        sentence = re.sub("\( *\,? *\)", "", sentence)
        sentence = re.sub("\( *[;,]", "(", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence

    def _create_examples(self, lines, set_type, negative_samples = 0):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            #print (line)
            text_a = convert_to_unicode(line[0])
            text_a = self.process_sent(text_a)
            title = convert_to_unicode(line[1])
            title = self.process_wiki_title(title)
            text_b = convert_to_unicode(line[2])
            text_b = self.process_evid(text_b)
            text_b = title + " : " + text_b
            label = convert_to_unicode(line[3])

            #print (text_a)
            #print (text_b)
            #print (line[2])
            #input("")
            if label == "2":
                if negative_samples != 0:
                    if random.randint(1,negative_samples) != 1:
                        continue
                    else:
                        label = 0
            else:
                label = 1
            #print (text_a, text_b, label)
            #input("")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples



class SecondSentenceRetrievalProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, filename=None):
        """See base class."""
        if filename is not None:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "train", 2)
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "sentence_retrieval_2_training_set.tsv")))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "sentence_retrieval_2_training_set.tsv")), "train", 2)


    def get_dev_examples(self, data_dir, filename=None):
        """See base class."""
        if filename is not None:        
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "dev", 0)
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "sentence_retrieval_2_dev_set.tsv")))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "sentence_retrieval_2_dev_set.tsv")), "dev", 0)

    def get_labels(self):
        """See base class."""
        return [None]


    def process_evid(self, sentence):
        sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
        sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
        sentence = re.sub("-LRB-", "(", sentence)
        sentence = re.sub("-RRB-", ")", sentence)
        sentence = re.sub("-COLON-", ":", sentence)
        sentence = re.sub("_", " ", sentence)
        sentence = re.sub("\( *\,? *\)", "", sentence)
        sentence = re.sub("\( *[;,]", "(", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence



    def _create_examples(self, lines, set_type, negative_samples = 0):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(line[0])
            text_a = self.process_evid(text_a)
            text_b = convert_to_unicode(line[2])
            text_b = self.process_evid(text_b)
            label = convert_to_unicode(line[3])
            if label == "2":
                if random.randint(1,negative_samples) != 1:
                    continue
                label = 0

            else:
                label = 1

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RTEProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def process_evid(self, sentence):
        sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
        sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
        sentence = re.sub("-LRB-", "(", sentence)
        sentence = re.sub("-RRB-", ")", sentence)
        sentence = re.sub("-COLON-", ":", sentence)
        sentence = re.sub("_", " ", sentence)
        sentence = re.sub("\( *\,? *\)", "", sentence)
        sentence = re.sub("\( *[;,]", "(", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence


    def get_train_examples(self, data_dir, filename=None):
        """See base class."""
        if filename is not None:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "train")
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "RTE_train_set.tsv")), "train")

    def get_dev_examples(self, data_dir, filename=None):
        """See base class."""
        if filename is not None:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, filename)), "dev")
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "RTE_dev_set.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            label = convert_to_unicode(line[2])
            #text_a = self.process_sent(text_a)
            text_b = self.process_evid(text_b)
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

fever_processors = {
    "first_sentence_retrieval": FirstSentenceRetrievalProcessor,
    "second_sentence_retriaval": SecondSentenceRetrievalProcessor,
    "rte": RTEProcessor,
}

fever_tasks_num_labels = {
    "first_sentence_retrieval": 1,
    "second_sentence_retriaval": 1,
    "rte": 3,
}

fever_output_modes = {
    "first_sentence_retrieval": "regression",
    "second_sentence_retrieval": "regression",
    "rte": "classification",
}
glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

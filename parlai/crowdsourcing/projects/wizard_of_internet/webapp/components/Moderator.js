/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Main exported functions in this file (valid_utterance and valid_search_query)
// are simple checks on the validity of user input (text or search query).
// There might also be additional checks on the submitted input in ParlAI world.

const STOPWORDS = [
  "",
  "''",
  "'d",
  "'ll",
  "'m",
  "'re",
  "'s",
  "'ve",
  "*",
  ",",
  "--",
  ".",
  "?",
  "``",
  "a",
  "about",
  "above",
  "after",
  "again",
  "against",
  "ain",
  "all",
  "also",
  "am",
  "an",
  "and",
  "any",
  "are",
  "aren",
  "as",
  "at",
  "be",
  "because",
  "been",
  "before",
  "being",
  "below",
  "between",
  "both",
  "but",
  "by",
  "can",
  "couldn",
  "d",
  "did",
  "didn",
  "do",
  "does",
  "doesn",
  "doing",
  "don",
  "down",
  "during",
  "each",
  "few",
  "for",
  "from",
  "further",
  "had",
  "hadn",
  "has",
  "hasn",
  "have",
  "haven",
  "having",
  "he",
  "her",
  "here",
  "hers",
  "herself",
  "him",
  "himself",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "isn",
  "it",
  "its",
  "itself",
  "just",
  "know",
  "ll",
  "m",
  "ma",
  "me",
  "mightn",
  "more",
  "most",
  "mustn",
  "my",
  "myself",
  "n't",
  "needn",
  "no",
  "nor",
  "not",
  "now",
  "o",
  "of",
  "off",
  "on",
  "once",
  "only",
  "or",
  "other",
  "our",
  "ours",
  "ourselves",
  "out",
  "over",
  "own",
  "people",
  "re",
  "really",
  "s",
  "same",
  "see",
  "shan",
  "she",
  "should",
  "shouldn",
  "so",
  "some",
  "such",
  "t",
  "than",
  "that",
  "the",
  "their",
  "theirs",
  "them",
  "themselves",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "through",
  "to",
  "too",
  "under",
  "until",
  "up",
  "ve",
  "very",
  "want",
  "was",
  "wasn",
  "we",
  "were",
  "weren",
  "what",
  "when",
  "where",
  "which",
  "while",
  "who",
  "whom",
  "why",
  "will",
  "with",
  "won",
  "wouldn",
  "y",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves"
];

// MIN_TEXT_LENGTH_TO_CHECK_COPY must always be less than OVERLAP_LENGTH_CHECK
// otherwise the check for copy/paste always passes
const MIN_TEXT_LENGTH_TO_CHECK_COPY = 20;
const OVERLAP_LENGTH_CHECK = 30;
const MIN_OVERLAP_REQUIRED = 1;
const MIN_NUM_WORDS_PER_UTTERANCE = 5;

const GREETING_FAREWELL_WORDS = ["hi", "hello", "bye", "goodbye"];

function split_tokenize(text) {
  const res = text.replace(/[.|. . .|,|;|:|!|?|(|)]/g, function(x) {
    return ` ${x} `;
  });
  return res.split(" ").filter(w => w !== "");
}

export default function valid_utterance(
  text,
  search_results,
  selected_results,
  isOnboarding,
  taskConfig
) {
  const bWords = taskConfig.bannedWords;
  const lowered_text = text.toLowerCase();
  return !(
    is_too_short(lowered_text, isOnboarding) ||
    looks_like_greetings(lowered_text, isOnboarding) ||
    has_did_you_know(lowered_text) ||
    has_banned_words(lowered_text, bWords) ||
    is_copy_pasted(lowered_text, search_results) ||
    has_turker_words(lowered_text) ||
    needs_more_overlap_with_selected(
      lowered_text,
      search_results,
      selected_results
    )
  );
}

export function valid_search_query(search_query, taskConfig) {
  const bWords = taskConfig.bannedWords;
  const lowered_search_query = search_query.toLowerCase();
  return !has_banned_words(lowered_search_query, bWords);
}

function is_too_short(text, isOnboarding) {
  if (isOnboarding) {
    return false;
  }

  const tokenized_text = split_tokenize(text);
  if (tokenized_text.length < MIN_NUM_WORDS_PER_UTTERANCE) {
    alert(
      "Your message was too short. Please try again and use longer and more engaging messages."
    );
    return true;
  }
  return false;
}

function looks_like_greetings(text, isOnboarding) {
  if (isOnboarding) {
    return false;
  }
  const first_word = split_tokenize(text)[0];
  if (GREETING_FAREWELL_WORDS.includes(first_word)) {
    alert(
      "Your message looks like a greeting or farewell. Please try again and use more engaging messages."
    );
    return true;
  }
  return false;
}

function has_did_you_know(text) {
  if (text.includes("did you know") || text.includes("did u know")) {
    alert(
      "Please try to be more engaging, and not use the phrase 'did you know' :)."
    );
    return true;
  }
  return false;
}

function has_turker_words(text) {
  if (text.includes("turker") || text.includes("turk")) {
    return !confirm(
      "Please do not mention the mechanical turk task in the conversation." +
        'Press "Cancel", to go back and edit, if your message does that, or "OK" to send the message.'
    );
  }
  return false;
}

function has_banned_words(text, banned_words_list) {
  const tokenized_text = split_tokenize(text);

  // Checking for banned words
  const banned_words = tokenized_text.filter(
    w => banned_words_list.indexOf(w) !== -1
  );
  if (banned_words.length > 0) {
    const detected_banned_words = banned_words.join(", ");
    alert(
      'We have detected the following offensive/banned language in your message: "' +
        detected_banned_words +
        '". Please edit and send again.'
    );
    return true;
  }
  return false;
}

function is_copy_pasted(text, docs) {
  if (
    !docs ||
    docs.length === 0 ||
    text.length < MIN_TEXT_LENGTH_TO_CHECK_COPY
  ) {
    return false;
  }

  function too_much_char_overlap(check_sentence, source_snetence) {
    const n = check_sentence.length;
    for (var s = 0; s < n; s += OVERLAP_LENGTH_CHECK) {
      const e = Math.min(n, s + OVERLAP_LENGTH_CHECK);
      if (e - s < MIN_TEXT_LENGTH_TO_CHECK_COPY) {
        continue;
      }
      const small_substr = check_sentence.substring(s, e);
      if (source_snetence.includes(small_substr)) {
        return true;
      }
    }
    return false;
  }

  for (var doc_id in docs) {
    const document = docs[doc_id];
    for (var sentence_id in document.content) {
      const sentence = document.content[sentence_id].toLocaleLowerCase();
      if (too_much_char_overlap(text, sentence)) {
        alert(
          "Your message has too much overlap with one of the candidates sentences. " +
            "Please retry and avoid copying and pasting."
        );
        return true;
      }
    }
  }
  return false;
}

function needs_more_overlap_with_selected(text, docs, selected) {
  if (
    !docs ||
    docs.length === 0 ||
    !selected ||
    selected.length === 0 ||
    selected[0][0] === true
  ) {
    return false;
  }

  var selected_sentences = [];
  for (var doc_id = 1; doc_id < selected.length; doc_id++) {
    const docSelection = selected[doc_id];
    for (var sent_id = 0; sent_id < docSelection.length; sent_id++) {
      if (docSelection[sent_id] === false) {
        // This sentence was not selected
        continue;
      }
      // "doc_id-1" because selection has an extra value at index 0 (nothing selected)
      const lower_selected_sentence = docs[doc_id - 1].content[
        sent_id
      ].toLowerCase();
      selected_sentences.push(lower_selected_sentence);
    }
  }
  const num_overlaps = overlap_number(selected_sentences.join(" "), text);
  if (MIN_OVERLAP_REQUIRED >= num_overlaps) {
    return !confirm(
      "Are you sure you are using the right checked sentence? " +
        "We have detected a lack of similarity between the checked sentence and your message " +
        '(please press "OK" if you intended to send this message, ' +
        'or "Cancel" to go back for edit).'
    );
  }
  return false;
}

function overlap_number(selected_sentences, text) {
  const PREFIX_LENGTH = 4;
  function reduce_to_prefix(s) {
    return split_tokenize(s)
      .filter(w => STOPWORDS.indexOf(w) === -1)
      .map(w => w.slice(0, PREFIX_LENGTH));
  }

  const text_tokens = reduce_to_prefix(text);
  const sentence_token = reduce_to_prefix(selected_sentences);
  return text_tokens.filter(word => sentence_token.indexOf(word) !== -1).length;
}

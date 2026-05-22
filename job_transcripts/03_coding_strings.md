---
interview_id: 03_coding_strings
participant: Anthony Chan (candidate, self-recorded)
role: Coding interview — longest substring without repeating characters
interview_type: technical
date: 2026-04-14
---

INTERVIEWER: Find the longest substring of a string that contains no repeating characters. Return its length.

CANDIDATE: Okay. So a brute force would be every substring and check uniqueness. That's O(n cubed) or O(n squared) with a set. The better approach is sliding window with a set. Two pointers, left and right. Right moves forward, adds the character to the set. If the character is already in the set, move left forward and remove characters from the set until the duplicate is gone. Track the max window size.

INTERVIEWER: Walk me through it on "abcabcbb".

CANDIDATE: Left zero, right at zero, set is empty. Add 'a', set is {a}, max is 1. Right moves to 1, add 'b', set is {a,b}, max is 2. Right moves to 2, add 'c', set is {a,b,c}, max is 3. Right moves to 3, 'a' is already in set. So remove 'a' from set, left moves to 1. Now set is {b,c}, add 'a', set is {a,b,c}, max stays 3. Right moves to 4, 'b' in set, remove 'b', left moves to 2, set is {a,c}, add 'b', set is {a,b,c}, max stays 3. And so on. Final max is 3.

INTERVIEWER: Good. Let's code it.

CANDIDATE: def lengthOfLongestSubstring(s): seen = set(); left = 0; best = 0; for right in range(len(s)): while s[right] in seen: seen.remove(s[left]); left += 1; seen.add(s[right]); best = max(best, right - left + 1); return best

INTERVIEWER: What about empty string?

CANDIDATE: Empty string returns zero because the for loop doesn't execute and best is initialized to zero. So that works.

INTERVIEWER: What about a single character?

CANDIDATE: 'a' — for loop runs once, right=0, while loop doesn't fire, seen becomes {a}, best becomes max(0, 0-0+1) = 1. Returns 1. Correct.

INTERVIEWER: Time and space complexity?

CANDIDATE: Time is O(n) because left and right each move at most n times. Space is O(min(n, alphabet_size)) for the set.

INTERVIEWER: One more thing — what if the input isn't a string, like None?

CANDIDATE: Oh. Yeah I didn't handle that. Should add at the top: if not s: return 0. That handles None and empty string both, since len(None) would throw and len('') is zero. Actually I'd be more explicit: if s is None: return 0. Then the empty string case is already handled by the for loop.

INTERVIEWER: Why didn't you catch that earlier?

CANDIDATE: Honestly because I focused on the algorithm and didn't think about input validation. That's a habit I need to build — always ask about input constraints before writing code. None inputs, very long strings, unicode, all things I didn't ask about.

INTERVIEWER: Good self-awareness. Let's wrap.

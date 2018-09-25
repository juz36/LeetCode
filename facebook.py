import random, string, heapq
import collections
class Solution(object):
    class TreeNode():
        def __init__(self, x):
            self.val = x
            self.left = None
            self.right = None
    
    class ListNode():
        def __init__(self, val):
            self.val = val
    
    class TrieNode(object):
        def __init__(self):
            self.children = {}
            self.is_word = False

    class MedianFinder:
        # TODO:
        def __init__(self):
            self.heaps = [], []

        def addNum(self, num):
            small, large = self.heaps
            heapq.heappush(small, -heapq.heappushpop(large, num))
            if len(large) < len(small):
                heapq.heappush(large, -heapq.heappop(small))

        def findMedian(self):
            small, large = self.heaps
            if len(large) > len(small):
                return float(large[0])
            return (large[0] - small[0]) / 2.0

    def addBinary(self, a, b):
        """LeetCode 67 Add Binary"""
        # Any input is 0
        if len(a)==0: 
            return b
        if len(b)==0:
            return a
        # Use recursive approach to search the rest of binaries
        if a[-1] == '1' and b[-1] == '1':
            # Tricky part
            return self.addBinary(self.addBinary(a[0:-1],b[0:-1]),'1')+'0'
        if a[-1] == '0' and b[-1] == '0':
            return self.addBinary(a[0:-1],b[0:-1])+'0'
        else:
            return self.addBinary(a[0:-1],b[0:-1])+'1'

    def addTwoNumbers(self, l1, l2):
        """LeetCode 2 Add two numbers"""
        # Use dummy head to link to the read head
        head = self.ListNode(0)
        cur = head
        # Use sum and carry to calculate the in-time result
        node_sum = 0
        node_carry = 0

        while l1 and l2:
            # Continuously add them
            node_sum = l1.val + l2.val + node_carry
            node_digit = node_sum % 10
            node_carry = node_sum / 10
            # After calculating the value, create new next node for the current, then move forward all
            cur.next = self.ListNode(node_digit)
            cur = cur.next
            l1 = l1.next
            l2 = l2.next
        # Come to final process
        tail = None
        if l1:
            tail = l1
        if l2:
            tail = l2
        # If the tail is 9 and carry is one, add new node for that
        while tail:
            if tail.val == 9 and node_carry == 1:
                cur.next = self.ListNode(0)
                node_carry = 1
            else:
                cur.next = self.ListNode(tail.val + node_carry) 
                node_carry = 0
            cur = cur.next
            tail = tail.next
        # Decide if we left the final tail
        if node_carry:
            cur.next = self.ListNode(node_carry)
        # Return the dummy.next
        return head.next

    class Codec:
        """LeetCode 535 Encode and Decode TinyURL"""
        def __init__(self):
            # Random of alphabetic
            self.alphabet = string.ascii_letters + '0123456789'
            self.url2code = {}
            self.code2url = {}

        def encode(self, longUrl):
            # If the longURL is not successly assigned, we will just go into loop
            while longUrl not in self.url2code:
                code = ''.join(random.choice(self.alphabet) for _ in range(6))
                if code not in self.code2url:
                    self.code2url[code] = longUrl
                    self.url2code[longUrl] = code
            return 'http://tinyurl.com/' + self.url2code[longUrl]

        def decode(self, shortUrl):
            # Get the url from the shortURL and search in code2url
            return self.code2url[shortUrl[-6:]]
    
    def findMedianSortedArrays(self, A, B):
        """LeetCode
        """
        # TODO:
        def kth(a, b, k):
            # If any array is empty, we directly return the non-empty one
            if not a:
                return b[k]
            if not b:
                return a[k]
            ia, ib = len(a) // 2 , len(b) // 2
            ma, mb = a[ia], b[ib]
            
            # when k is bigger than the sum of a and b's median indices 
            if ia + ib < k:
                # if a's median is bigger than b's, b's first half doesn't include k
                if ma > mb:
                    return kth(a, b[ib + 1:], k - ib - 1)
                else:
                    return kth(a[ia + 1:], b, k - ia - 1)
            # when k is smaller than the sum of a and b's indices
            else:
                # if a's median is bigger than b's, a's second half doesn't include k
                if ma > mb:
                    return kth(a[:ia], b, k)
                else:
                    return kth(a, b[:ib], k)
        l = len(A) + len(B)
        if l % 2 == 1:
            return kth(A, B, l // 2)
        else:
            return (kth(A, B, l // 2) + kth(A, B, l // 2 - 1)) / 2.   

    class WordDictionary(object):
        def __init__(self):
            self.root = TrieNode()
        
        def addWord(self, word):
            # Create TrieNode from root
            node = self.root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.is_word = True
            

        def search(self, word):
            return self.searchFrom(self.root, word)
        
        def searchFrom(self, node, word):
            # For each character, we see if there are any path and end-point
            for i in range(len(word)):
                c = word[i]
                if c == '.':
                    for k in node.children:
                        if self.searchFrom(node.children[k], word[i+1:]):
                            return True
                    return False
                elif c not in node.children:
                    return False
                node = node.children[c]
            return node.is_word

    def findKthLargest(self, nums, k):
        """LeetCode 215 Kth Largest Element in an Array
        """
        # TODO:
        # Use max heap to store the k-largest numbers
        import heapq
        count = []
        for num in nums:
            heapq.heappush(count, num)
            # if the length is larger than k, we pop the smallest one
            if len(count) > k:
                heapq.heappop(count)
        return heapq.heappop(count)

    def searchRotatedSortedArray(self, nums, target):
        # TODO:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) / 2
            if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]):
                lo = mid + 1
            else:
                hi = mid
        return lo if target in nums[lo:lo+1] else -1

    def longestConsecutive(self, root):
        """LeetCode 298 Binary Tree Longest Consecutive Sequence"""
        if not root:
            return 0
        
        ret = 0
        # The structure that store the (node, count) value
        stack = [(root, 1)]
        while stack:
            node, cnt = stack.pop()
            # If any side of binary is consecutive, plus the count, and append it to stack
            if node.left:
                stack.append((node.left, cnt+1 if node.left.val == node.val + 1 else 1))
            if node.right:
                stack.append((node.right, cnt+1 if node.right.val == node.val + 1 else 1))
            # Get the maximum count
            ret = max(ret, cnt)
            
        return ret

    def isSubtree(self, s, t):
        """LeetCode Subtree of another Tree"""
        # Compare if two nodes are the same
        def isMatch(s, t):
            if not(s and t):
                return s is t
            return (s.val == t.val and 
                    isMatch(s.left, t.left) and 
                    isMatch(s.right, t.right))
        # If both trees are same
        if isMatch(s, t): 
            return True
        # if we reach the end, return False
        if not s: return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def buildTree2(self, preorder, inorder):
        """LeetCode 
        """
        if inorder:
            # Get the root from preorder
            ind = inorder.index(preorder.pop(0))
            # Create the order and set it as root
            root = self.TreeNode(inorder[ind])
            # Move from left to right to construct the rest
            root.left = self.buildTree(preorder, inorder[0:ind])
            root.right = self.buildTree(preorder, inorder[ind+1:])
            return root

    def buildTree(self, inorder, postorder):
        """LeetCode 106 Construct binary tree from in-order and post-order
        """
        if not inorder or not postorder:
            return None
        # Get the root element of the tree from post-order
        root = self.TreeNode(postorder.pop())
        inorderIndex = inorder.index(root.val)
        # Get the left and right tree of the root from in-order
        root.right = self.buildTree(inorder[inorderIndex+1:], postorder)
        root.left = self.buildTree(inorder[:inorderIndex], postorder)

        return root

    # Bold/Focus on impact/Move fast/Be open/Build social value
    def productExceptSelf(self, nums):
        """LeetCode 238 Product of Array Except Self: 
        https://leetcode.com/problems/product-of-array-except-self/description/
        Time: O(n)
        Space: O(1)
        """
        # Product the array with 1 offset
        res = [0] * len(nums)
        res[0] = 1
        for i in range(1, len(nums)):
            res[i] = res[i-1] * nums[i-1]
        
        # Now reverse the order with one offset
        tmp = 1
        for i in reversed(range(len(nums))):
            res[i] *= tmp
            tmp *= nums[i]
        return res

    def findCelebrity(self, n):
        """LeetCode: 277 Find the celebrity
        https://leetcode.com/problems/find-the-celebrity/description/
        
        """
        # dummy function to ignore error
        def knows(a, b):
            pass
        # first round find the candidate
        candidate = 0
        for i in range(1, n):
            if knows(candidate, i):
                candidate = i
        
        for i in range(n):
            if i != candidate and ( knows(candidate, i) or not knows(i, candidate)):
                return -1
        return candidate
    
    def intersection(self, nums1, nums2):
        """
        LeetCode 349 Intersection of two Arrays
        This is the DP version, 
        O(nlogn) time complexity
        """
        def binary_search(nums, num):
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = lo + (hi-lo)//2
                if nums[mid] == num:
                    return True
                elif nums[mid] > num:
                    hi = mid - 1
                elif nums[mid] < num:
                    lo = mid + 1
            return False
        if not nums1 or not nums2:
            return []
        
        res = set()
        nums1.sort()
        # Sort and binary search the element O(nlogn)
        for num in nums2:
            test = binary_search(nums1, num)
            print(test)
            if test:
                res.add(num)
        return list(res)
    
    def intToRoman(self, num):
        M = ["", "M", "MM", "MMM"]
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        # Fast approach
        return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10]

    def romanToInt(self, s):
        """LeetCode 13 Roman to Integer"""
        res = 0
        conversion = {"I": 1, "V": 5, "X": 10, "L": 50,"C": 100, "D": 500, "M": 1000}
        size = len(s)
        i = 0
        while i < size:
            if i < size - 1 and conversion[s[i+1]] > conversion[s[i]]:
                res += conversion[s[i+1]] - conversion[s[i]]
                i += 1
            else:
                res += conversion[s[i]]
            i += 1
        return res

    def calculator(self, s):
        """LeetCode 224: Basic Calculator"""
        total = 0
        i, signs = 0, [1, 1]
        while i < len(s):
            c = s[i]
            # If the character is digit, get the following digits
            if c.isdigit():
                start = i
                while i < len(s) and s[i].isdigit():
                    i += 1
                total += signs.pop() * int(s[start:i])
                continue
            # If the character is +-(, we can assign -1 for -, + for both (
            if c in '+-(':
                signs += signs[-1] * (1, -1)[c == '-']
            # if the character is ), pop the left (
            elif c == ')':
                signs.pop()
            # Continue the loop until i reach end
            i += 1
        return total

    def flatten(self, head):
        # class Node(val, prev, next, child):
        """LeetCode 430 Flatten a Multilevel Doubly Linked List"""
        if not head:
            return
        
        dummy = object() # Node(0, None, head, None)
        stack = []
        stack.append(head)
        prev = dummy
        while stack:
            # Connect previous node with current node
            root = stack.pop()
            root.prev = prev
            prev.next = root
            # first append next node to stack
            if root.next:
                stack.append(root.next)
                root.next = None
            # Then append child nodes to stack
            if root.child:
                stack.append(root.child)
                root.child = None
            prev = root
        # Final process to remove the dummy
        dummy.next.prev = None
        return dummy.next

    def pick(self, nums, target):
        """LeetCode 398"""
        # Reservoir Sampling
        import random
        res = None
        count = 0
        for i, x in enumerate(nums):
            if x == target:
                # The new item has 1/i probability to replace res
                count += 1
                chance = random.randint(1, count)
                if chance == count:
                    res = i
        return res

    def addOperators(self, num, target):
        """LeetCode 282 Expression Operator"""
        def dfs(num, temp, cur, last, res):
            if not num:
                if cur == target:
                    res.append(temp)
                return
            for i in range(1, len(num)+1):
                val = num[:i]
                # prevent "00*" as a number
                if i == 1 or (i > 1 and num[0] != "0"): 
                    dfs(num[i:], temp + "+" + val, cur+int(val), int(val), res)
                    dfs(num[i:], temp + "-" + val, cur-int(val), -int(val), res)
                    dfs(num[i:], temp + "*" + val, cur-last+last*int(val), last*int(val), res)
        res, target = [], target
        for i in range(1, len(num)+1):
            # prevent "00*" as a number
            if i == 1 or (i > 1 and num[0] != "0"): 
                # this step put first number in the string
                dfs(num[i:], num[:i], int(num[:i]), int(num[:i]), res) 
        return res

    def merge_interval(self, intervals):
        pass

    def summary_range(self, nums):
        """
        LeetCode: 228
        time: O(n)
        space: O(n)
        """
        res = []
        if not nums or len(nums) == 0:
            return nums
        
        i = 0
        while i < len(nums):
            num = nums[i]
            # Move forward until we meet a gap
            while i < len(nums) - 1 and nums[i] + 1 == nums[i+1]:
                i += 1
            # if the current num is not destination num
            if num != nums[i]:
                # append them together
                res.append("{0}->{1}".format(num, nums[i]))
            else:
                # Add it to result
                res.append(str(num))
            i += 1

        return res

    def top_k_frequent_elements(self, nums, k):
        """LeetCode 347
        time
        space
        """
        import heapq
        from collections import Counter
        cnt = Counter(nums)
        freqs = []
        heapq.heapify(freqs)
        for num, count in cnt.items():
            heapq.heappush(freqs, (count, num))
            if len(freqs) > k:
                heapq.heappop(freqs)
        
        res = []
        for _ in range(k):
            res.append(heapq.heappop(freqs)[1])
        return res

    def top_k_frequent(self, words, k):
        """LeetCode 692"""
        class Element:
            def __init__(self, count, word):
                self.count = count
                self.word = word
            
            def __lt__(self, other):
                if self.count == other.count:
                    return self.word > other.word
                return self.count < other.count
            
            def __eq__(self, other):
                return self.count == other.count and self.word == other.word

        # Count word frequency
        import heapq
        from collections import Counter
        cnt = Counter(words)
        # Add each entry to a min heap with k max elements
        freq = []
        heapq.heapify(freq)
        for word, count in cnt.items():
            heapq.heappush(freq, ((Element(count, word), word)) )
            if len(freq) > k:
                heapq.heappop(freq)
        # Reverse the heap
        res = []
        for _ in range(k):
            res.append(heapq.heappop(freq)[1])
        return res[::-1]

    def top_k_frequent2(self, words):
        # 1. use hash to count frequency
        # 2. sort the counter
        # cannot satisfy < O(nlogn) and O(n) space
        pass

    def numberToWords(self, num):
        """LeetCode 273
        time:
        space:
        """
        thousands = ['', 'Thousand', 'Million', 'Billion']
        tens = ['', 'ten', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        less_than_20 = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 
                'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
        def helper(num):
            if num < 20:
                return less_than_20[num]
            elif num < 100:
                return tens[num // 10] + " " + less_than_20[num % 10]
            else:
                return ' '.join([less_than_20[num // 100], 'Hundred', helper(num % 100)])
        count = 0
        result = ''
        if num == 0:
            return 'Zero'
        while num > 0:
            if num % 1000 == 0:
                count += 1
                num //= 1000
                continue
            result = ' '.join([helper(num % 1000).strip(), thousands[count], result]).strip()
            count += 1
            num //= 1000
        return result.strip()

    def isNumber(self, s):
        """LeetCode 65
        time: 
        space:
        """
        import re
        # Match the number before e
        number = r"^(0?|[1-9][0-9]*)(\.[0-9]*[1-9])?$"
        # number = "^[-+]?([0-9]+(\.[0-9]*)?)$"
        is_num = lambda x : re.match(number, x) != None
        # match the number after e
        is_int = lambda x : re.match("^[-+]?[0-9][0-9]*$", x) != None
        s = s.strip().lower()
        s = s.split('e')
        if len(s) > 2:
            return False
        return is_num(s[0]) and (is_int(s[1]) if len(s) == 2 else True)

    def find_closest_elements(self, arr, k, x):
        """LeetCode 658
        time: O(logn + k)
        space: O(1)
        """
        from bisect import bisect_right
        # Find the right position of arr
        index = bisect_right(arr, x)
        i = index - 1
        j = index
        # Search the left and right
        while k > 0:
            k -= 1
            if i < 0 or (j < len(arr) and abs(arr[i] - x) > abs(arr[j]-x)):
                j += 1
            else:
                i -= 1
        return arr[i+1:j]

    def smallestDistancePair(self, nums, k):
        """LeetCode 791 """
        nums.sort()
        N = nums[-1]
        count = [0] * (N+1)
        
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                count[nums[j] - nums[i]] += 1
        
        for i in range(N):
            k -= count[i]
            if k <= 0:
                return i
        return 0
        

    def diameterOfBinaryTree(self, root):
        """LeetCode 543"""
        self.res = 0
        def helper(root):
            if root is None:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            self.res = max(self.res, left + right + 1)
            return 1 + max(left, right)
        
        if not root:
            return 0
        
        helper(root)
        return self.res - 1

    def increasing_triplet_subsequence(self, nums):
        """LeetCode 334"""
        MAX = 2 ** 32
        first, second = MAX, MAX
        for num in nums:
            # If both number are not initiated
            if num <= first:
                first = num
            # if any number is smaller than first max
            elif num <= second:
                second = num
            # if any number is smaller than both first and second max
            else:
                return True
        return False
    
    def multiply(self, num1, num2):
        #placeholder for multiplication ndigit by mdigit result in n+m digits
        product = [0] * (len(num1) + len(num2)) 
        # position within the placeholder
        position = len(product)-1 

        for n1 in num1[::-1]:
            tempPos = position 
            for n2 in num2[::-1]:
                # adding the results of single multiplication
                product[tempPos] += int(n1) * int(n2) 
                # bring out carry number to the left array
                product[tempPos-1] += product[tempPos]//10 
                # remove the carry out from the current array
                product[tempPos] %= 10 
                # first shifting the multplication to the end of the first integer
                tempPos -= 1 
            # then once first integer is exhausted shifting the second integer and starting 
            position -= 1 

        # once the second integer is exhausted we want to make sure we are not zero padding  
        pointer = 0 # pointer moves through the digit array and locate where the zero padding finishes
        while pointer < len(product)-1 and product[pointer] == 0: # if we have zero before the numbers shift the pointer to the right
            pointer += 1

        return ''.join(map(str, product[pointer:])) # only report the digits to the right side of the pointer

    def lengthOfLIS(self, nums):
        """LeetCode 300 Longest Increasing Subsequence"""
        # DP version
        if not nums:
            return 0
        if len(nums) == 1:
            return 1
        dp = [1] * len(nums)
        dp[0] = 1
        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
    def lengthOfLIS2(self, nums):
        # Binary Search version
        tails = [0] * len(nums)
        size = 0
        # use bisect.bisect_left is also OK
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) // 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size

    def move_zeros(self, nums):
        # LeetCode 283 Move Zeros
        if not nums or len(nums) == 1:
            return
        
        slow = 0
        # fill the front of array with non-zero numbers
        for i, number in enumerate(nums):
            if number != 0:
                nums[slow] = number
                slow += 1
        # If the pointer is by the end of the array, the following loop won't proceed
        for i in range(slow, len(nums)):
            nums[i] = 0
            

    def minWindow(self, s, t):
        # LeetCode 76 Minimum window substring: high frequency
        if len(s) < len(t):
            return ""
        from collections import Counter
        MAX = 2 ** 32
        res_len = MAX
        hashtb = Counter(t)
        match_count = 0
        left, right = 0, 0
        index = 0
        
        for right, c in enumerate(s):
            if c in hashtb:
                hashtb[c] -= 1
                if hashtb[c] == 0:
                    match_count += 1
            while match_count == len(hashtb):
                if res_len > right - left + 1:
                    res_len = right - left + 1
                    index = left
                left_most = s[left]
                if left_most in hashtb:
                    hashtb[left_most] += 1
                    if hashtb[left_most] == 1:
                        match_count -= 1
                left += 1
        if res_len == MAX:
            return ""
        return s[index: index + res_len]

    def minWindow2(self, s, t):
        cnt = [0] * 128
        for character in s:
            c = ord(character)
            cnt[c] += 1
        
        MAX = 2 ** 32
        start = 0
        total = len(t)
        res_len = MAX

        left = 0
        for right in range(len(s)):
            i = ord(s[right])
            if cnt[i] > 0:
                total -= 1
            cnt[i] -= 1

            while total == 0:
                j = ord(s[left])
                if res_len > right - left + 1:
                    res_len = right - left + 1
                    start = left
                cnt[j] += 1
                if cnt[j] > 0:
                    total += 1
                left += 1
        if res_len == MAX:
            return ""
        return s[start, start + res_len]

        

    def removeInvalidParentheses(self, s):
        # LeetCode 301
        def isValid(s):
            s = ' '.join(filter('()'.count, s))
            while '()' in s:
                s = s.replace('()', '')
                return not s
        level = {s}
        while True:
            # Use filter to judge if the level is right
            valid = list(filter(isValid, level))
            if valid:
                return valid
            # Get all the possible output from level
            level = {s[:i] + s[i+1:] for s in level for i in range(len(s))}

    def threeSum(self, nums):
        """
        LeetCode 15 3Sum
        """
        result_array = []
        nums.sort()
        
        for index in range(len(nums)-2):
            if nums[index] == nums[index-1] and index > 0:
                continue
            # Start from index + 1 to end of array
            left, right = index + 1, len(nums)-1
            while left < right:
                # If sum left + right + index is not 0
                s = nums[left] + nums[right] + nums[index]
                if s<0:
                    left += 1
                elif s>0:
                    right -= 1
                else:
                    # Else remove add it to result and move duplicate element
                    result_array.append([nums[index], nums[left], nums[right]])
                    while left < right and nums[left+1] == nums[left]:
                        left += 1
                    while left < right and nums[right-1] == nums[right]:
                        right -= 1
                    left += 1
                    right -= 1
                        
            return result_array
    
    def climbStairs(self, n):
        # LeetCode 40
        if n == 0 or n == 1:
            return n
        
        dp = [0] * (n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

    def isBipartite(self, graph):
        """LeetCode 785 Is Graph Bipartite
        Bipartite define that there are only two color for all nodes, and no edge between same colored node
        """

        color = {}

        def dfs(pos):
            for i in graph[pos]:
                # If the node is painted
                if i in color:
                    if color[i] == color[pos]:
                        return False
                else:
                    # Color any adjacent node to another color
                    color[i] = 1 - color[pos]
                    if not dfs(i):
                        return False
            return True
        for i in range(len(graph)):
            if i not in color:
                # Paint with 0 color
                color[i] = 0
            if not dfs(i):
                return False
        return True

    def isBipartite2(self, graph):
        size = len(graph)
        visited = [0] * size

        for i in range(size):
            if graph[i] and visited[i] == 0:
                visited[i] = 1
                q = []
                q.append(i)
                while q:
                    cur = q.pop()
                    for c in graph[cur]:
                        if visited[c] == 0:
                            visited[c] = 3 - visited[i]
                            q.append(c)
                        else:
                            if visited[c] == visited[cur]:
                                return False
        return True
            
    
    def merge(self, nums1, m, nums2, n):
        """LeetCode 88 Merge Sorted Array"""
        # Start from the end of array, and move forward to start while adding the current maximum value to index
        for index in reversed(range(m+n)):
            if m > 0 and n > 0:
                if nums1[m-1] > nums2[n-1]:
                    nums1[index] = nums1[m-1]
                    m -= 1
                else:
                    nums1[index] = nums2[n-1]
                    n -= 1
            else:
                break
        
        if n > 0:
            for index in range(0, n):
                nums1[index] = nums2[index]
    
    def canAttendMeetings(self, intervals):
        """meeting room 252"""
        if not intervals or len(intervals) == 1:
            return True
        start = []
        end = []
        for interval in intervals:
            start.append(interval[0])
            end.append(interval[1])
        start.sort()
        end.sort()
        for i in range(1, len(intervals)):
            if end[i-1] > start[i]:
                return False
        return True
    
    def minMeetingRooms(self, intervals):
        """253 Meeting Rooms II"""
        res = 0
        if not intervals:
            return 0
        if len(intervals) == 1:
            return 1
        
        starts = []
        ends = []
        for interval in intervals:
            starts.append(interval[0])
            ends.append(interval[1])
        # Sort the start and end array
        starts.sort()
        ends.sort()

        end = 0
        # Move the end or add result
        for i in range(len(starts)):
            if starts[i] < ends[end]:
                res += 1
            else:
                end += 1
        return res

    def minMeetingRooms2(self, intervals):
        """253 Meeting Rooms II"""
        import heapq
        intervals = list(sorted(intervals, key= lambda x: x.start))
        heap = []
        for i in intervals:
            if heap and heap[0] <= i.start:
                heapq.heapreplace(heap, i.end)
            else:
                heapq.heappush(heap,i.end)
        return len(heap)

        


solution = Solution()
# solution.top_k_frequent(['I', 'love', 'I', 'love', 'LeetCode', 'coding'], 2)
solution.lengthOfLIS2([10,9,2,5,3,7,101,18])
# solution.canAttendMeetings([[7, 10],[2, 4]])
# print(solution.minWindow("ADOBECODEBANC", "ABC"))

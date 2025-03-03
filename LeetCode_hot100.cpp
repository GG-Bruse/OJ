/*
����֮��
https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
����һ����������nums��һ������Ŀ��ֵtarget,�����ڸ��������ҳ���ΪĿ��ֵtarget������������
���������ǵ������±�
����Լ���ÿ������ֻ���Ӧһ����,�����㲻��ʹ��������ͬ��Ԫ��
����԰�����˳�򷵻ش�
*/
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target)
    {
        unordered_map<int, int> hashMap;
        for (int i = 0; i < nums.size(); ++i)
        {
            unordered_map<int, int>::iterator it = hashMap.find(target - nums[i]);
            if (it != hashMap.end()) return { i, it->second };
            else hashMap[nums[i]] = i;
        }
        return {};
    }
};



/*
��ĸ��λ�ʷ���
https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
����һ���ַ������飬���㽫��ĸ��λ�������һ�𡣿��԰�����˳�򷵻ؽ���б�
��ĸ��λ��������������Դ���ʵ�������ĸ�õ���һ���µ���
����: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
���: [["bat"],["nat","tan"],["ate","eat","tea"]]
*/
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs)
    {
        unordered_map<string, vector<string>> hashMap;
        for (string& str : strs) {
            string tmp = str;
            sort(tmp.begin(), tmp.end());
            hashMap[tmp].push_back(str);
        }
        vector<vector<string>> result;
        for (auto it = hashMap.begin(); it != hashMap.end(); ++it)
            result.push_back(it->second);
        return result;
    }
};



/*
����г���
https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked
����һ��δ������������� nums ���ҳ���������������У���Ҫ������Ԫ����ԭ�������������ĳ���
������Ʋ�ʵ��ʱ�临�Ӷ�Ϊ O(n) ���㷨���������

���룺nums = [100,4,200,1,3,2]
�����4
���ͣ���������������� [1, 2, 3, 4]�����ĳ���Ϊ 4
*/
#include <iostream>
#include <unordered_set>
using namespace std;
class Solution
{
public:
    int longestConsecutive(vector<int>& nums)
    {
        unordered_set<int> hashSet;
        for (size_t i = 0; i < nums.size(); ++i)
            hashSet.insert(nums[i]);

        int longest = 0;
        for (auto& number : hashSet)
        {
            if (hashSet.count(number - 1)) continue;
            else
            {
                int currentNumber = number;
                int length = 1;
                while (hashSet.count(currentNumber + 1)) {
                    currentNumber += 1;
                    length += 1;
                }
                longest = max(longest, length);
                if (hashSet.size() == longest) break;
            }
        }
        return longest;
    }
};



/*
�ƶ���
https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked
����һ������ nums����дһ������������ 0 �ƶ��������ĩβ��ͬʱ���ַ���Ԫ�ص����˳��
��ע�⣬�����ڲ���������������ԭ�ض�������в���
*/
#include <iostream>
#include <vector>
using namespace std;
class Solution
{
public:
    void moveZeroes(vector<int>& nums)
    {
        int start = 0, end = 0;
        while (end < nums.size())
        {
            if (0 != nums[end]) {
                swap(nums[start], nums[end]);
                ++start;
            }
            ++end;
        }
    }
};



/*
ʢˮ��������
https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked
����һ������Ϊ n ���������� height ���� n �����ߣ��� i ���ߵ������˵��� (i, 0) �� (i, height[i])
�ҳ����е������ߣ�ʹ�������� x �Ṳͬ���ɵ�����������������ˮ
�����������Դ�������ˮ��
˵�����㲻����б����
*/
#include <iostream>
#include <vector>
using namespace std;
class Solution
{
public:
    // ÿ���ƶ�����Ϊȡ�����ˮ, ���̰岻��, ȡ��ˮ��Զ�������һ�θ���
    int maxArea(vector<int>& height)
    {
        int left = 0, right = height.size() - 1;
        int result = 0;
        while (left < right)
        {
            int area = min(height[left], height[right]) * (right - left);
            result = max(area, result);
            if (height[left] <= height[right]) ++left;
            else --right;
        }
        return result;
    }
};



/*
����֮��
https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums ���ж��Ƿ������Ԫ�� [nums[i], nums[j], nums[k]] ���� i != j��i != k �� j != k
ͬʱ������ nums[i] + nums[j] + nums[k] == 0
���㷵�����к�Ϊ 0 �Ҳ��ظ�����Ԫ�顣
ע�⣺���в����԰����ظ�����Ԫ��
*/
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class Solution
{
public:
    vector<vector<int>> threeSum(vector<int>& nums)
    {
        int size = nums.size();
        sort(nums.begin(), nums.end()); // ���ڲ����ظ�, ���������
        vector<vector<int>> result;
        for (size_t i = 0; i < size - 2; ++i)
        {
            if (i > 0 && nums[i] == nums[i - 1]) continue;//������һ��ö�ٵ�������ͬ
            int target = 0 - nums[i];
            int k = size - 1;
            for (int j = i + 1; j < size - 1; ++j)
            {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;//������һ��ö�ٵ�������ͬ
                while (j < k && nums[j] + nums[k] > target) --k;// ˫ָ�����, <��=��ֹͣ
                // nums[j] < ����whileö�ٹ���nums[k], ��nums[j]����, nums[k]���С
                // ����j == k��, ˵���������е�k, ����nums[j] + nums[k] > target
                // j�������ֻ���ܸ���, ���Ҳ������и�С��nums[k]��
                // ��ʱ��������j��ѭ�� 
                if (j == k) break;
                if (nums[j] + nums[k] == target)
                    result.push_back({ nums[i], nums[j], nums[k] });
            }
        }
        return result;
    }
};



/*
���ظ��ַ�����Ӵ�
https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked
����һ���ַ���s �������ҳ����в������ظ��ַ�����Ӵ��ĳ���
����: s = "abcabcbb"
���: 3
����: ��Ϊ���ظ��ַ�����Ӵ��� "abc"�������䳤��Ϊ 3
*/
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    int lengthOfLongestSubstring(string s)
    {
        int size = s.size();
        if (0 == size) return 0;

        int maxLength = 1;
        int prev = 0, last = 0;
        unordered_map<char, int> hashCount;
        while (prev < size)
        {
            while (last < size && hashCount[s[last]] == 0) {
                hashCount[s[last]]++;
                ++last;
            }
            maxLength = max(maxLength, last - prev);
            hashCount[s[prev]]--;
            ++prev;
        }
        return maxLength;
    }
};



/*
�ҵ��ַ�����������ĸ��λ��
https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked
���������ַ��� s �� p���ҵ� s ������ p ����λ�ʵ��Ӵ���������Щ�Ӵ�����ʼ�����������Ǵ������˳��
����: s = "cbaebabacd", p = "abc"
���: [0,6]
����:
��ʼ�������� 0 ���Ӵ��� "cba", ���� "abc" ����λ��
��ʼ�������� 6 ���Ӵ��� "bac", ���� "abc" ����λ��
*/
// error : timeout
class Solution
{
public:
    vector<int> findAnagrams(string s, string p)
    {
        vector<int> result;
        unordered_map<char, int> hashExist;
        for (char& ch : p) hashExist[ch]++;

        int size = s.size();
        int prev = 0, last = 0;
        unordered_map<char, int> hashCurrent;
        while (prev < size)
        {
            while (last < size && hashCurrent[s[last]] < hashExist[s[last]]) {
                hashCurrent[s[last]]++;
                ++last;
            }
            bool isHeterotopicWords = true;
            for (auto& ch : p) {
                if (hashCurrent[ch] != hashExist[ch]) {
                    isHeterotopicWords = false;
                    break;
                }
            }
            if (isHeterotopicWords) result.push_back(prev);
            ++prev;
            last = prev;
            hashCurrent.clear();
        }
        return result;
    }
};
// ��������
#include <iostream>
#include <vector>
#include <string>
using namespace std;
class Solution
{
public:
    vector<int> findAnagrams(string s, string p)
    {
        int sSize = s.size(), pSize = p.size();
        if (sSize < pSize) return vector<int>();

        vector<int> result;
        vector<int> sHash(26);
        vector<int> pHash(26);
        for (int i = 0; i < pSize; ++i) {
            ++sHash[s[i] - 'a'];
            ++pHash[p[i] - 'a'];
        }

        if (sHash == pHash) result.push_back(0);

        for (int i = 0; i < sSize - pSize; ++i) {
            // ��󻬶�һ��
            --sHash[s[i] - 'a'];
            ++sHash[s[i + pSize] - 'a'];
            if (sHash == pHash) result.push_back(i + 1);
        }
        return result;
    }
};



/*
��Ϊ K ��������
https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums ��һ������ k ������ͳ�Ʋ����� �������к�Ϊ k ��������ĸ���
��������������Ԫ�ص������ǿ�����
���룺nums = [1,1,1], k = 2
�����2
*/
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    // pre[i]Ϊnums[0...i]��������֮��, pre[i] = pre[i - 1] + nums[i]
    //[i,j]��Ϊk������, ��prev[j] - prev[i - 1] == k, ����Ϊpre[i - 1] = prev[j] - k
    int subarraySum(vector<int>& nums, int k)
    {
        // ��¼prev[i]���ڸ���
        unordered_map<int, int> hashMap;
        hashMap[0] = 1;

        int result = 0;
        int pre = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            pre += nums[i];// �������ӵ�prev[j]
            // �Ƿ����prev[j] - k, ��pre[i - 1]
            if (hashMap.find(pre - k) != hashMap.end())
                result += hashMap[pre - k];
            ++hashMap[pre];
        }
        return result;
    }
};



/*
�����������ֵ
https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums����һ����СΪ k �Ļ������ڴ������������ƶ�����������Ҳ�
��ֻ���Կ����ڻ��������ڵ� k �����֡���������ÿ��ֻ�����ƶ�һλ
���� ���������е����ֵ
���룺nums = [1,3,-1,-3,5,3,6,7], k = 3
�����[3,3,5,5,6,7]
*/
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
class Solution
{
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k)
    {
        int size = nums.size();
        priority_queue<pair<int, int>> qe;
        for (int i = 0; i < k; ++i)
            qe.emplace(nums[i], i);
        vector<int> result = { qe.top().first };

        for (int i = k; i < size; ++i) //�����ƶ��Ĵ����ұ߽�
        {
            qe.emplace(nums[i], i);
            while (qe.top().second <= i - k) qe.pop();//���Ѷ�Ԫ�ز��ڻ�����������ɾ��
            result.push_back(qe.top().first);
        }
        return result;
    }
};



/*
��С�����Ӵ�
https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked
����һ���ַ��� s ��һ���ַ��� t
���� s �к��� t �����ַ�����С�Ӵ������ s �в����ں��� t �����ַ����Ӵ����򷵻ؿ��ַ��� ""
ע��:
���� t ���ظ��ַ�������Ѱ�ҵ����ַ����и��ַ��������벻���� t �и��ַ�����
��� s �д����������Ӵ������Ǳ�֤����Ψһ�Ĵ�
���룺s = "ADOBECODEBANC", t = "ABC"
�����"BANC"
���ͣ���С�����Ӵ� "BANC" ���������ַ��� t �� 'A'��'B' �� 'C'
*/
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    bool check() // ��鵱ǰ�����Ƿ����t����������ַ�
    {
        for (const auto& it : origin)
            if (count[it.first] < it.second)
                return false;
        return true;
    }

    string minWindow(string s, string t)
    {
        for (const char& ch : t) ++origin[ch];

        int left = 0, right = -1;
        int resultLeft = -1, resultRight = -1;
        int length = INT_MAX;
        int size = s.size();

        while (right < size) //�ұ߽粻������
        {
            // �ұ߽�ָ����ַ���������t��, count���м�¼
            if (origin.find(s[++right]) != origin.end()) ++count[s[right]];
            // ������߽�, ����Ҫ���ǰ����, ԽСԽ��
            while (check() && left <= right)
            {
                if (right - left + 1 < length) { //�޸�result
                    length = right - left + 1;
                    resultLeft = left;
                }
                if (origin.find(s[left]) != origin.end()) //ɾ��count�еļ�¼
                    --count[s[left]];
                ++left;
            }
        }
        return resultLeft == -1 ? string() : s.substr(resultLeft, length);
    }
private:
    unordered_map<char, int> origin, count;
    // origin ���t�и���Ԫ�س��ֵĴ���
    // count ά����ǰ���������и���Ԫ�س��ֵĴ���
};



/*
�������������
https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums �������ҳ�һ���������͵����������飨���������ٰ���һ��Ԫ�أ�������������
�������������е�һ����������
���룺nums = [-2,1,-3,4,-1,2,1,-5,4]
�����6
���ͣ����������� [4,-1,2,1] �ĺ����Ϊ 6 ��
*/
class Solution
{
public:
    int maxSubArray(vector<int>& nums)
    {
        int size = nums.size();
        int result = nums[0];
        // dp[i] : ��nums[i]Ϊ��β������� ���������������
        vector<int> dp(size);
        dp[0] = nums[0];
        for (int i = 1; i < size; ++i) {
            dp[i] = max(dp[i - 1] + nums[i], nums[i]);
            result = max(result, dp[i]);
        }
        return result;
    }
};
// �ռ��Ż�
class Solution
{
public:
    int maxSubArray(vector<int>& nums)
    {
        int size = nums.size();
        int result = nums[0];
        int prev = nums[0];
        for (int i = 1; i < size; ++i) {
            prev = max(prev + nums[i], nums[i]);
            result = max(result, prev);
        }
        return result;
    }
};



/*
�ϲ�����
https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked
������ intervals ��ʾ���ɸ�����ļ��ϣ����е�������Ϊ intervals[i] = [starti, endi]
����ϲ������ص������䣬������ һ�����ص����������飬��������ǡ�ø��������е���������
*/
class Solution
{
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals)
    {
        // sort(intervals.begin(), intervals.end(), [](const vector<int>& val1, const vector<int>& val2){
        //     return val1[0] < val2[0];
        // });
        // ��дlambda,Ĭ�ϰ��յ�һ��������
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        for (int i = 0; i < intervals.size(); ++i)
        {
            int left = intervals[i][0], right = intervals[i][1];
            if (!merged.size() || merged.back()[1] < left)
                merged.push_back({ left, right });
            else
                merged.back()[1] = max(merged.back()[1], right);
        }
        return merged;
    }
};



/*
��ת����
https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums���������е�Ԫ��������ת k ��λ�ã����� k �ǷǸ���
����: nums = [1,2,3,4,5,6,7], k = 3
���: [5,6,7,1,2,3,4]
����:
������ת 1 ��: [7,1,2,3,4,5,6]
������ת 2 ��: [6,7,1,2,3,4,5]
������ת 3 ��: [5,6,7,1,2,3,4]
*/
class Solution
{
public:
    void rotate(vector<int>& nums, int k)
    {
        if (k == 0) return;
        int size = nums.size();
        vector<int> result(size);
        for (int i = 0; i < size; ++i)
            result[(i + k) % size] = nums[i];
        nums.assign(result.begin(), result.end());
    }
};



/*
��������������ĳ˻�
https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums������ ���� answer ������ answer[i] ���� nums �г� nums[i] ֮�������Ԫ�صĳ˻�
��Ŀ���� ��֤ ���� nums֮������Ԫ�ص�ȫ��ǰ׺Ԫ�غͺ�׺�ĳ˻�����  32 λ ������Χ��
�벻Ҫʹ�ó��������� O(n) ʱ�临�Ӷ�����ɴ���
����: nums = [1,2,3,4]
���: [24,12,8,6]
*/
class Solution
{
public:
    vector<int> productExceptSelf(vector<int>& nums)
    {
        int size = nums.size();
        vector<int> prevMulSum(size), lastMulSum(size);
        prevMulSum[0] = 1;
        for (int i = 1; i < size; ++i)
            prevMulSum[i] = prevMulSum[i - 1] * nums[i - 1];
        lastMulSum[size - 1] = 1;
        for (int i = size - 2; i >= 0; --i)
            lastMulSum[i] = lastMulSum[i + 1] * nums[i + 1];

        vector<int> result(size);
        for (int i = 0; i < size; ++i)
            result[i] = prevMulSum[i] * lastMulSum[i];
        return result;
    }
};
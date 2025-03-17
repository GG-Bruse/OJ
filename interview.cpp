#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
using namespace std;






/*
给定一个二叉树，返回该二叉树层序遍历的结果，（从左到右，一层一层地遍历）
给定的二叉树是{3,9,20,#,#,15,7}, 返回结果[[3],[9,20],[15,7]]
*/
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
class Solution 
{
public:
    vector<vector<int> > levelOrder(TreeNode* root)
    {
        if (root == nullptr) return {};

        vector<vector<int>> result;
        queue<TreeNode*> qe;
        qe.push(root);

        while (!qe.empty())
        {
            vector<int> layer;
            int size = qe.size();
            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = qe.front();
                layer.push_back(node->val);
                qe.pop();
                if (node->left != nullptr) qe.push(node->left);
                if (node->right != nullptr) qe.push(node->right);
            }
            result.push_back(layer);
        }
        return result;
    }
};
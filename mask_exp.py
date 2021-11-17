    #这段代码之前嵌在模型内部，如果需要改成独立于模型的函数，可以将h,h_w作为模型计算得到的参数传入。
    def mask_exp(self,batch):
        batch = [b.cuda() for b in batch]

        inputs = batch[0:10]
        length = inputs[0].size(1)    #num of tokens
        self.model.eval()
        logits,h,_ = self.model(inputs)      #conventional procedure   size of h:(1,50)
        h = h.squeeze(0).cpu().detach().numpy()   #convert to numpy h为正常得分（所有词都没有mask）

        h_w = list(range(length))
        r = [0.00 for _ in range(length)]    #score

        for i in range(length):
           _,h_w[i],adj=self.model(inputs,flag=True,mask_pos=i)   #h_w[i]是第i个词被mask掉的得分
           h_w[i] = h_w[i].squeeze(0).cpu().detach().numpy()
           for dim in range(len(h)):
              r[i] += abs(h[dim]-h_w[i][dim])

        max_r = max(r)

        r = [r[i]/max_r for i in range(length)]

        print(r)
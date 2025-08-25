import motmetrics as mm

gt   = mm.io.loadtxt('eval file/ground_truth.txt', fmt='mot15-2D', min_confidence=1)
pred = mm.io.loadtxt('eval file/results_mot.txt', fmt='mot15-2D') 

gt.index   = gt.index.set_names(['FrameId', 'Id'])
pred.index = pred.index.set_names(['FrameId', 'Id'])
    
gt   = gt.sort_index()
pred = pred.sort_index()

acc = mm.utils.compare_to_groundtruth(gt, pred, 'iou', distth=0.5)
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames','mota','motp','idf1','precision','recall'],
                     name='bytetrack')
print(mm.io.render_summary(summary, formatters=mh.formatters,
                           namemap=mm.io.motchallenge_metric_names))

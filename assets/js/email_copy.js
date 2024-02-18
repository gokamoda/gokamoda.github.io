function email_copy(){
    var email = "go.kamoda@dc.tohoku.ac.jp";
    if(navigator.clipboard) {
        navigator.clipboard.writeText(email).then(function() {
          alert('Email address copied')
        });
      } else {
          alert('Copy failed. Email address is go.kamoda@dc.tohoku.ac.jp');
      }
};

cite_map = {
  "kamoda-etal-2023-test": '@inproceedings{kamoda-etal-2023-test,\n\
    title = "Test-time Augmentation for Factual Probing",\n\
    author = "Kamoda, Go  and Heinzerling, Benjamin  and Sakaguchi, Keisuke  and Inui, Kentaro",\n\
    editor = "Bouamor, Houda  and Pino, Juan  and Bali, Kalika",\n\
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",\n\
    month = dec,\n\
    year = "2023",\n\
    address = "Singapore",\n\
    publisher = "Association for Computational Linguistics",\n\
    url = "https://aclanthology.org/2023.findings-emnlp.236",\n\
    pages = "3650--3661",\n\
    abstract = "Factual probing is a method that uses prompts to test if a language model knows certain world knowledge facts. A problem in factual probing is that small changes to the prompt can lead to large changes in model output. Previous work aimed to alleviate this problem by optimizing prompts via text mining or fine-tuning. However, such approaches are relation-specific and do not generalize to unseen relation types. Here, we propose to use test-time augmentation (TTA) as a relation-agnostic method for reducing sensitivity to prompt variations by automatically augmenting and ensembling prompts at test time. Experiments show improved model calibration, i.e., with TTA, model confidence better reflects prediction accuracy. Improvements in prediction accuracy are observed for some models, but for other models, TTA leads to degradation. Error analysis identifies the difficulty of producing high-quality prompt variations as the main challenge for TTA.",\n\
    }'
}
function cite_copy(key){
  if(navigator.clipboard) {
      navigator.clipboard.writeText(cite_map[key]).then(function() {
        alert('Copied')
      });
    } else {
        alert('Copy failed. BibTeX is ' + cite_map[key]);
    }
}
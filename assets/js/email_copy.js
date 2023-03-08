function email_copy(){
    var email = "go.kamoda@dc.tohoku.ac.jp";
    if(navigator.clipboard) {
        navigator.clipboard.writeText(email).then(function() {
          alert('email copied')
        });
      } else {
          alert('対応していません。');
      }
};
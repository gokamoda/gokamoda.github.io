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
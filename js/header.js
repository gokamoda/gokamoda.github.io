function header(){
    console.log(1)
    $.ajax({
        url: './template/header.html',
        cache: false,
        async: false,
        dataType: 'html',
        success: function(html){
            document.write(html);
        }
    });
}
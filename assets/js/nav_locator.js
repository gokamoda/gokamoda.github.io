var pastOffsetTop = 0
var presentOffsetTop = 0
var futureOffsetTop = 0

function focus(id){
    var element = document.getElementsByName(id)
    for (var i = 0; i < element.length; i++) {
        element[i].style.marginLeft = "10px";
    }
    // element1.style.marginLeft = "10px";
}
function unfocus(id){
    var element1 = document.getElementsByName(id)
    for (var i = 0; i < element1.length; i++) {
        element1[i].style.marginLeft = "0px";
    }
}

window.onscroll = function (e) {
    var verticalPosition = 0;
    if (scrollY)//usual
        verticalPosition = scrollY;
    else if (document.documentElement.clientHeight)//ie
        verticalPosition = document.documentElement.scrollTop;
    else if (document.body)//ie quirks
        verticalPosition = document.body.scrollTop;


        
        
    var halfPageHeight = window.innerHeight / 2;
    var pastThreshold = pastOffsetTop - halfPageHeight;
    var presentThreshold = presentOffsetTop - halfPageHeight;
    var futureThreshold = futureOffsetTop - halfPageHeight;
    if (verticalPosition < pastThreshold) {
        unfocus('nav_past')
        unfocus('nav_present')
        unfocus('nav_future')
        // console.log("unfocus all")
    } else if(verticalPosition >= pastThreshold && verticalPosition < presentThreshold){
        focus('nav_past')
        unfocus('nav_present')
        unfocus('nav_future')
        // console.log("focus past")
    }else if(verticalPosition >= presentThreshold && verticalPosition < futureThreshold){
        unfocus('nav_past')
        focus('nav_present')
        unfocus('nav_future')
        // console.log("focus present")
    }else if(verticalPosition >= futureThreshold){
        unfocus('nav_past')
        unfocus('nav_present')
        focus('nav_future')
        // console.log("focus future")
    }
}

window.onload = function (e) {
    element_past = document.getElementById('past')
    pastOffsetTop = element_past.offsetTop

    element_present = document.getElementById('present')
    presentOffsetTop = element_present.offsetTop

    element_future = document.getElementById('future')
    futureOffsetTop = element_future.offsetTop
}



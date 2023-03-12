function focus(id){
    // var element = document.getElementById(id);
    var element1 = document.getElementsByName(id)[0]
    var element2 = document.getElementsByName(id)[1]
    element1.style.marginLeft = "20px";
    element2.style.marginLeft = "15px";
}
function unfocus(id){
    // var element = document.getElementById(id);
    var element1 = document.getElementsByName(id)[0]
    var element2 = document.getElementsByName(id)[1]
    element1.style.marginLeft = "-20px";
    element2.style.marginLeft = "0px";
}

window.onscroll = function (e) {
    var vertical_position = 0;
    if (scrollY)//usual
        vertical_position = scrollY;
    else if (document.documentElement.clientHeight)//ie
        vertical_position = document.documentElement.scrollTop;
    else if (document.body)//ie quirks
        vertical_position = document.body.scrollTop;



    var page_height = window.innerHeight;
    var offset = page_height / 2
    if (vertical_position < page_height - offset) {
        unfocus('nav_past')
        unfocus('nav_present')
        unfocus('nav_future')
    } else if(vertical_position >= page_height - offset && vertical_position < page_height * 2 - offset){
        focus('nav_past')
        unfocus('nav_present')
        unfocus('nav_future')
    }else if(vertical_position >= page_height * 2 - offset && vertical_position < page_height * 3 - offset){
        unfocus('nav_past')
        focus('nav_present')
        unfocus('nav_future')
    }else if(vertical_position >= page_height * 3 - offset && vertical_position < page_height * 4 - offset){
        unfocus('nav_past')
        unfocus('nav_present')
        focus('nav_future')
    }
    // var logo_div = document.getElementById('logo');
    // var right_div = document.getElementsByClassName('col_right')[0];
    // var scrollbarWidth = window.innerWidth - document.body.clientWidth;

    // // console.log(window.innerWidth)
    // // console.log(window.innerWidth / 4)

    // var init_pos_top = window.innerHeight * 0.23;
    // var init_pos_right = window.innerWidth * 0.5;
    // var init_height = window.innerHeight * 0.17;
    // var init_width = init_height * 3;

    // var final_width = document.body.clientWidth / 4;
    // var final_height = final_width * 0.5;
    // var final_pos_top = 0;
    // var final_pos_right = final_width / 2;



    // if (final_width > init_width) {
    //     var get_bigger = true
    // }else{
    //     var get_bigger = false
    // }

    // var range = init_pos_top;

    // var after_pos_top = init_pos_top + (final_pos_top - init_pos_top) * (vertical_position / range);
    // var after_pos_right = init_pos_right + (final_pos_right - init_pos_right) * (vertical_position / range);
    // var after_width = init_width + (final_width - init_width) * (vertical_position / range);
    // var after_height = init_height + (final_height - init_height) * (vertical_position / range);

    // if (after_pos_top > final_pos_top) {
    //     logo_div.style.top = after_pos_top + 'px';
    // } else {
    //     logo_div.style.top = final_pos_top + 'px';
    // }

    // if (after_pos_right > final_pos_right) {
    //     logo_div.style.right = after_pos_right + 'px';
    // } else {
    //     logo_div.style.right = final_pos_right + 'px';
    // }

    // if (get_bigger){
    //     if (after_width < final_width) {
    //         logo_div.style.width = after_width + 'px';
    //     } else {
    //         logo_div.style.width = final_width + 'px';
    //     }
    // }else{
    //     if (after_width > final_width) {
    //         logo_div.style.width = after_width + 'px';
    //     } else {
    //         logo_div.style.width = final_width + 'px';
    //     }
    // }

}

/*
begin:
  top: 20vh
  right: 50%
  height: 20vh
  width: 10vh
end:
  top: 0
  right: 0 - 5%
  width: 25%
  height: 20hw
*/




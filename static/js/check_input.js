function predictButton(){
    let title = document.getElementById("title").value;
    let content = document.getElementById("content").value;
    if(title == ''){
        alert("請輸入文章標題");
    }
    else if(content == ''){
        alert("請輸入文章內容");
    }
    else{
        alert("預測文章中，請耐心等候>_<");
    }
}

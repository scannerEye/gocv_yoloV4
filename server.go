package main

import (
	"log"
	"net/http"
	"jlmx/server"
)

func main() {

	// public views
	http.HandleFunc("/", server.HandleIndex)
	http.HandleFunc("/admin/file/saveFile", server.SaveFile)
	http.HandleFunc("/admin/algorithm/algTest", server.AlgTestFunc)

	log.Fatal(http.ListenAndServe(":9191", nil))
}


##### Setup environment for data processing #####
rm(list=ls())
library(dplyr)
library(survival)
library(gtsummary)

##### Load Visit Data #####
long_raw_name <- read.table("longitudinal_master_pheno_new.txt", fill = TRUE, nrows=1) %>%
  select(-V1) %>%
  unlist %>%
  unname
long_raw <- read.table("longitudinal_master_pheno_new.txt", header=F, fill = TRUE, skip=1)
names(long_raw) <- tolower(long_raw_name)
# 4757
# 79594    32

# reference file
photo0 <- read.delim('master_pheno_jpg.txt') # dim: 187996,47 #subjects: 4628
names(photo0) <- tolower(names(photo0))
photo0$eye <- toupper(photo0$eye)
photo0$side <- toupper(photo0$side)
photo0$angle <- toupper(photo0$angle)
# only analyze F2 angle LS fundus images
photo <- photo0 %>%
  filter((angle=='F2') & (side == 'LS') & !(img_name  %in% c('54250_14_F2_LE_LS.jpg',
                                                             '55518_16_F2_RE_LS.jpg',
                                                             '54341_10_F2_LE_LS.jpg',
                                                             '54265_QUA_F2_LE_LS.jpg'))) %>%
  mutate(visno = replace(visno, visno == "BL", "00")) %>%
  mutate(visno = as.numeric(replace(visno, visno == "00", "0")))
# delete 4 observations. There is no corresponding imgs
# 75110 47

# landmark, use 55 years' old as time zero
time_landmark <- round(min(photo$enrollage)) # 55

##### Merge Photo and Visit data ####
long_tmp <- long_raw %>%
  mutate(uni_visit_id = paste(id2,visno,eye,sep="_"))

# Same plot from different batches (2010 vs 2014)
# Use the 2014 image if the visit has images from both batches
photo_tmp <- photo %>%
  mutate(uni_visit_id = paste(id2,visno,eye,sep="_")) %>%
  group_by(uni_visit_id) %>%
  filter(!((n()>1) & (batch==2010))) %>%
  ungroup

# 57 images are not related to any visit record
# photo_tmp[!(photo_tmp$uni_visit_id %in% long_tmp$uni_visit_id),] %>% View
# Delete those photos

df0 <- merge(long_tmp,
             photo_tmp %>% select(img_name,uni_visit_id,rc_id,batch,sampleid,amdcat,marital,school,occupat,smokedyn,smkagest,smkpacks,smkcurr,smknocig,smkageqt,bphighyn,bpmednow),
             by="uni_visit_id",
             all.x=T,suffixes = c("",".photo"))
# select variables which are unique in photo_tmp
# 79594 49
# write.csv(df0,file="Data/long_photo.csv",row.names = F)

##### Data preparation ####
df <- df0 %>%
  subset(!is.na(amdsev)) %>%
  arrange(id2,eye,visno) %>%
  group_by(id2,eye) %>%
  # define clinic variables
  mutate(high_school = ifelse(school>2,1,0),
         smoke_current = ifelse((smokedyn=='Y' & smkcurr=='Y'),1,0),
         smoke_former = ifelse((smokedyn=='Y' & smkcurr=='N'),1,0),
         smoke_never = ifelse((smokedyn=='N'),1,0),
         amdsev_baseline = dplyr::first(amdsev),
         amdsev_lead1 = dplyr::lead(amdsev, n = 1, default = tail(amdsev,1)),
         amdsev_last = tail(amdsev,1)) %>%
  # focus on left side
  # filter out the observations with label "cannot grade"
  filter((amdsev_baseline <9) &
           (amdsev-amdsev_lead1 < 6) # delete those visit with AMDSEV decreased from 11 to 1
  ) %>%
  mutate(
    cant_grade = 0,
    cant_grade = ifelse(drszwi == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(rpedwi == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(drsoft == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(ssrf2 == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(drcalc == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(drreti %in% c(2,8), cant_grade+1, cant_grade),
    cant_grade = ifelse(drarwi == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(incpwi >= 7, cant_grade+1, cant_grade),
    cant_grade = ifelse(geoawi >= 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(subff2 == 8, cant_grade+1, cant_grade),
    cant_grade = ifelse(subhf2 == 8, cant_grade+1, cant_grade),
    drszwi = ifelse(drszwi == 8, 0, drszwi),
    rpedwi = ifelse(rpedwi == 8, 0, rpedwi),
    drsoft = ifelse(drsoft == 8, 0, drsoft),
    ssrf2 = ifelse(ssrf2 == 8, 0, ssrf2),
    drcalc = ifelse(drcalc == 8, 0, drcalc),
    drreti = ifelse(drreti %in% c(2,8), 0, drreti),
    drarwi = ifelse(drarwi == 8, 0, drarwi),
    incpwi = ifelse(incpwi >= 7, 0, incpwi),
    geoawi = ifelse(geoawi >= 8, 0, geoawi),
    subff2 = ifelse(subff2 == 8, 0, subff2),
    subhf2 = ifelse(subhf2 == 8, 0, subhf2),
  ) %>%
  mutate(ssrf2 = ifelse(ssrf2>1,1,0),
         drcalc = ifelse(drcalc>1,1,0),
         drreti = ifelse(drreti>1,1,0),
         geoawi = ifelse(geoawi>2,1,0),
         subff2 = ifelse(subff2>1,1,0),
         subhf2 = ifelse(subhf2>1,1,0),) %>%
  # define amdsev and tstop before dropiing observations
  mutate(amdsev_lead2 = dplyr::lead(amdsev, n = 1, default = tail(amdsev,1)),
         tstop1 = dplyr::lead(phtime, n = 1, default = 999)) %>%
  filter(!is.na(img_name)) %>%    # deletes visits without images
  filter(cant_grade<1) %>%    # deletes images with low quality
  
  # delete we don't have "baseline" img (not gradable) but already late-AMD
  mutate(amdsev_baseline2 = dplyr::first(amdsev)) %>%
  filter(amdsev_baseline2<9) %>%
  
  mutate(amdsev_lead3 = dplyr::lead(amdsev, n = 1, default = tail(amdsev,1)),
         t0 = first(phtime),
         tstart = phtime,
         tstop2 = dplyr::lead(phtime, n = 1, default = 999),
         # tstop = ifelse(tstop1 == 999,tstop2,tstop1),
         tstop = ifelse(tstop2 == 999,tstop1,tstop2),
         event = ifelse(pmax(amdsev_lead2,amdsev_lead3)>8,1,0),
         
         # first event
         min_event_tstart0 = ifelse(event == 1, tstart,999),
         # use min_event_time to select the first event
         min_event_tstart = min(min_event_tstart0),
         first_event = ifelse(tstart <= min_event_tstart,1,0)
  ) %>%
  mutate(addtime = enrollage-time_landmark,
         tstart55 = tstart + addtime,
         tstop55 = tstop + addtime) %>%
  filter((tstop < 999) & (first_event==1) & (tstart<tstop) & t0==0) %>%
  mutate(tstop55_final = max(tstop55),
         tstop_final = max(tstop)) %>%
  # id2 2444 and 2699 have duplicated phtime
  # "3994LE" and "3994LE" have wrong phtime
  filter(!(uni_visit_id %in% c("3994_14_RE","3994_14_LE"))) %>% 
  mutate(event = ifelse(uni_visit_id %in% c("2444_18_RE","2699_6_LE"),1,event),
         event_final = max(event),
         id2_eye = paste0(id2,eye),
         obsno = row_number(),
         drszwi_base = first(drszwi),
         rpedwi_base = first(rpedwi),
         drsoft_base = first(drsoft),
         drcalc_base = first(drcalc),
         drreti_base = first(drreti),
         drarwi_base = first(drarwi),
         incpwi_base = first(incpwi),
         
         drszwi_diff = drszwi-dplyr::lag(drszwi,default=first(drszwi)),
         rpedwi_diff = rpedwi-dplyr::lag(rpedwi,default=first(rpedwi)),
         drsoft_diff = drsoft-dplyr::lag(drsoft,default=first(drsoft)),
         drcalc_diff = drcalc-dplyr::lag(drcalc,default=first(drcalc)),
         drreti_diff = drreti-dplyr::lag(drreti,default=first(drreti)),
         drarwi_diff = drarwi-dplyr::lag(drarwi,default=first(drarwi)),
         incpwi_diff = incpwi-dplyr::lag(incpwi,default=first(incpwi)),
  ) %>%
  ungroup()

##### Number of Cases, Event rate and Overall Fitting ####
dim(df)
# 53076    91

##### Data splitting #####
set.seed(233)
id_all <- unique(df$id2) %>% as.character()

# Test data
id_test <- sample(id_all,length(id_all)*0.1)
df_test <- df %>% filter(id2 %in% id_test)
# (df_test$event %>% sum())/(df_test %>% select(id2,eye) %>% n_distinct)
# event rate 0.2249357
save(df_test,file='start_stop_test.Rdata')

# Train and Validation data
id_all2 <- setdiff(id_all,id_test)
id_tenfold <- floor(runif(length(id_all2), 1,11))
sampSizes <- sapply(seq(1:10), function(s){length(which(id_tenfold==s))})
sampSizes

df_working <- merge(df %>% filter(id2 %in% id_all2),
                    data.frame(id2=id_all2,id_tenfold=id_tenfold),by='id2')
table(df_working$id_tenfold)
# event rate for each data
# for(i in 1:10){print(sum(df_working %>% filter(id_tenfold==i) %>% .$event)/(df_working %>% filter(id_tenfold == i) %>% select(id2,eye) %>% n_distinct))}
# [1] 0.1842105
# [1] 0.178273
# [1] 0.1931973
# [1] 0.2171053
# [1] 0.229709
# [1] 0.2119013
# [1] 0.2049689
# [1] 0.2009063
# [1] 0.2162516
# [1] 0.1889339
save(df_working,file='start_stop_working.Rdata')
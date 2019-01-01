def train(data_loader, net, criterion, optimizer, epoch):
    net.train(True)
    train_loss = 0.
    for i, (inputs, labels) in enumerate(data_loader, 0):
        print("Training epoch %d: batch # %d" % (epoch, i), end='\r')
        # map to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(data_loader)

def validate(data_loader, net, criterion, measures, epoch):
    net.train(False)
    val_loss = 0.
    measurements = {k:0. for k in measures.keys()}
    for i, (inputs, labels) in enumerate(data_loader, 0):
        print("Validating epoch %d: batch # %d" % (epoch, i), end='\r')
        # map to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        for (k,m) in measures.items():
            measurements[k] += m(outputs, labels).item()
        
        val_loss += loss.item()
    
    for k in measures.keys():
        measurements[k] = measurements[k] / len(data_loader)
    return val_loss / len(data_loader), measurements

def fit(net, train_loader, val_loader, criterion, optimizer, lrscheduler, measures, epoch):
    train_loss = train(train_loader, net, criterion, optimizer, epoch)
    val_loss, measurements = validate(val_loader, net, criterion, measures, epoch)
    lrscheduler.step(val_loss)
    
    print("epoch {}: train-loss={:.5f}, val-loss={:.5f}, new_lr={:.5f}".format(epoch, train_loss, val_loss, optimizer.param_groups[0]['lr']))
    measurements['train_loss'] = train_loss
    measurements['val_loss'] = val_loss
    return measurements
